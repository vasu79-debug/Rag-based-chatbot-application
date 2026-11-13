import asyncio
import base64
import contextvars
import ipaddress
import json
import os
import re
import socket
import time
import logging
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import urlparse
from langchain_core.tools import tool
from langgraph.types import interrupt
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Playwright
from config import settings

logger = logging.getLogger(__name__)

AUDIT_LOG_FILE = "audit.log"
EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# --- Per-session sandbox state -------------------------------------------------
# One Playwright driver for the whole process; one ISOLATED headless browser
# (its own browser + context + page) per chat session_id. This is what makes
# each chat get its own "sandbox" instead of every user sharing one global
# browser/window.
_playwright: Optional[Playwright] = None
_sessions: Dict[str, dict] = {}  # session_id -> {"browser", "context", "page"}

# The LangGraph tools below are called by the LLM with plain args (url,
# selector, text, ...) - they never see a session_id. The agent graph sets
# this contextvar once per turn (see graph/mcp_graph.py) so the tool
# implementations know which sandbox to operate on. Contextvars propagate
# into any asyncio tasks created after the value is set, which covers how
# LangGraph executes tool calls.
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_session_id", default="default_session"
)
_current_user_prompt: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_prompt", default=""
)

# Sessions the user hit Emergency Stop on. While a session_id is in here,
# navigate() refuses to silently spin a brand-new sandbox back up for it -
# otherwise a still-running agent turn (the backend doesn't know it was
# stopped until its next tool call) would reopen the sandbox window right
# after the user just closed it. Cleared automatically the next time the
# user sends a new message for that session (see clear_kill_flag()).
_killed_sessions: set = set()


def set_current_session(session_id: str) -> None:
    """Call once per agent turn (from the graph) before invoking the tools below."""
    _current_session_id.set(session_id or "default_session")


def set_current_task_context(session_id: str, prompt: str = "") -> None:
    """Sets session ID and user task prompt for semantic guard evaluation."""
    _current_session_id.set(session_id or "default_session")
    if prompt:
        _current_user_prompt.set(prompt)


async def _ensure_playwright() -> Playwright:
    global _playwright
    if _playwright is None:
        _playwright = await async_playwright().start()
    return _playwright


async def init_browser(session_id: str) -> Page:
    """Create (or reuse) an isolated, headless sandbox for this session.

    Headless because this is no longer a window on the server's own desktop -
    the live view is streamed to the browser UI over /ws/sandbox/{session_id}
    instead, so the sandbox has to work the same way on a server with no
    display at all.
    """
    pw = await _ensure_playwright()
    sess = _sessions.get(session_id)
    if sess and not sess["page"].is_closed():
        return sess["page"]

    ws_endpoint = (getattr(settings, "BROWSER_WS_ENDPOINT", None) or os.getenv("BROWSER_WS_ENDPOINT", "")).strip()
    if not ws_endpoint:
        ws_endpoint = "ws://localhost:3000"

    browser = None
    if ws_endpoint:
        if "?" not in ws_endpoint:
            ws_endpoint += "?stealth=true"
        elif "stealth" not in ws_endpoint:
            ws_endpoint += "&stealth=true"
        try:
            logger.info(f"Connecting to Docker Sandbox Browser at {ws_endpoint} for session '{session_id}'...")
            browser = await pw.chromium.connect_over_cdp(ws_endpoint)
        except Exception as e:
            logger.warning(f"Could not connect to Docker browser at {ws_endpoint}: {e}. Falling back to local Chromium.")
            browser = None

    if not browser:
        logger.info(f"Launching local Playwright Chromium for session '{session_id}'...")
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ]
        )

    context = await browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        locale="en-US"
    )
    # Anti-bot detection stealth script: hides automation flags from Spotify, Cloudflare, Arkose Labs
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        window.chrome = { runtime: {}, loadTimes: function() {}, csi: function() {}, app: {} };
    """)
    page = await context.new_page()
    _sessions[session_id] = {
        "browser": browser,
        "context": context,
        "page": page,
        "last_active": time.time(),
        # Human-takeover state: while manual_control is True, the frontend
        # sandbox window forwards the user's own mouse/keyboard into this
        # page instead of the agent acting on it. See request_human_action().
        "manual_control": False,
        "manual_control_reason": "",
        "manual_control_details": "",
    }

    # Automatically track new tabs/popups opened by links (target="_blank" or window.open)
    def handle_popup_page(new_p):
        sess = _sessions.get(session_id)
        if sess:
            logger.info(f"Session '{session_id}': Website opened a new tab/popup. Tracking new active page.")
            sess["page"] = new_p
    context.on("page", handle_popup_page)

    logger.info(f"Sandbox browser started for session '{session_id}' (headless).")
    return page


def touch_session(session_id: Optional[str] = None) -> None:
    """Updates the last-active timestamp of a browser session to prevent idle reaping."""
    sid = session_id or _current_session_id.get()
    sess = _sessions.get(sid)
    if sess:
        sess["last_active"] = time.time()


async def get_page(session_id: Optional[str] = None) -> Optional[Page]:
    """Return the live page for a session without creating one. Used by the
    tools (via the current-session contextvar) and by the live-view websocket."""
    sid = session_id or _current_session_id.get()
    sess = _sessions.get(sid)
    if sess and not sess["page"].is_closed():
        sess["last_active"] = time.time()
        return sess["page"]
    return None


async def get_current_url(session_id: Optional[str] = None) -> Optional[str]:
    page = await get_page(session_id)
    return page.url if page else None


async def close_browser(session_id: Optional[str] = None) -> None:
    """Closes one session's sandbox, or every open sandbox when no session_id
    is given. (Internal cleanup helper - for the Emergency Stop button, use
    emergency_stop() below, which also halts the agent and blocks reopen.)"""
    targets = [session_id] if session_id else list(_sessions.keys())
    for sid in targets:
        sess = _sessions.pop(sid, None)
        if sess:
            try:
                await sess["browser"].close()
            except Exception as e:
                logger.warning(f"Error closing sandbox for session '{sid}': {e}")
            logger.info(f"Sandbox browser closed for session '{sid}'.")


IDLE_SESSION_TIMEOUT_SECONDS = int(os.getenv("BROWSER_IDLE_TIMEOUT_SECONDS", "600"))  # 10 minutes default


async def browser_session_reaper() -> None:
    """
    Background worker that runs periodically to close inactive browser sessions.
    Prevents lingering Chromium contexts and memory leaks on the host/Docker container.
    """
    logger.info(f"Browser session reaper started (idle timeout: {IDLE_SESSION_TIMEOUT_SECONDS}s).")
    while True:
        try:
            await asyncio.sleep(60)  # Check every 60 seconds
            now = time.time()
            idle_sessions = []

            for sid, sess in list(_sessions.items()):
                # Never reap while human takeover (manual control) is active
                if sess.get("manual_control"):
                    continue
                last_active = sess.get("last_active", now)
                if now - last_active > IDLE_SESSION_TIMEOUT_SECONDS:
                    idle_sessions.append(sid)

            for sid in idle_sessions:
                logger.info(f"Reaping idle browser session '{sid}' (inactive for > {IDLE_SESSION_TIMEOUT_SECONDS}s).")
                log_audit("IDLE_SESSION_REAPED", f"Session: {sid} | Inactive timeout reached")
                await close_browser(sid)

        except asyncio.CancelledError:
            logger.info("Browser session reaper stopped.")
            break
        except Exception as e:
            logger.error(f"Error in browser session reaper: {e}")



async def emergency_stop(session_id: str) -> None:
    """The actual Emergency Stop action: closes the sandbox AND marks the
    session so navigate() won't silently reopen a new one if the agent's
    turn is still finishing up server-side. main.py also cancels the
    in-flight agent task - this is the belt-and-suspenders half of that."""
    _killed_sessions.add(session_id)
    await close_browser(session_id)
    await exit_manual_control(session_id)


def is_killed(session_id: str) -> bool:
    return session_id in _killed_sessions


def clear_kill_flag(session_id: str) -> None:
    """Called at the start of every new agent turn so a stale stop from a
    previous turn doesn't block this one."""
    _killed_sessions.discard(session_id)


async def reset_page(session_id: str) -> None:
    """Called once at the start of a brand-new user message (not a resumed
    turn) so a fresh task never starts by showing whatever page a previous,
    unrelated task left open in the live sandbox view."""
    sess = _sessions.get(session_id)
    if not sess or sess["page"].is_closed():
        return
    if sess.get("manual_control"):
        # Never yank the page out from under a human who is actively using
        # it mid-takeover (a new message can arrive on the same session
        # while an earlier turn is still paused waiting on the user).
        return
    try:
        await sess["page"].goto("about:blank")
    except Exception as e:
        logger.warning(f"Could not reset page for session '{session_id}': {e}")


# --- Domain blocklist (open navigation by default) ---------------------------
# ChatGPT Agent and Claude for Chrome don't pre-approve a list of sites -
# they leave browsing open and block only specific risk categories by
# default, gating RISKY ACTIONS (payments, purchases, data submission) with
# human approval instead. This mirrors that: navigate() reaches any site
# unless its domain (or a parent of it) is explicitly listed here. The
# file is re-read on every check, so an edit takes effect immediately.
BLOCKED_DOMAINS_FILE = os.path.join(os.path.dirname(__file__), "blocked_domains.json")


def _load_blocked_domains() -> list:
    try:
        with open(BLOCKED_DOMAINS_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {BLOCKED_DOMAINS_FILE}: {e}")
        return []


FORBIDDEN_CATEGORY_KEYWORDS = [
    "casino", "bet365", "poker", "porn", "xxx", "adult", "torrent",
    "gamble", "roulette", "lottery", "warez", "darkweb"
]


def _is_ip_private(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
    except ValueError:
        return False


def validate_navigation_security(url: str) -> tuple[bool, str]:
    """
    Validates URL safety before Playwright navigates:
    1. Only http and https schemes are permitted.
    2. Blocks internal/private hostnames & IPs (SSRF protection).
    3. Blocks unauthorized content categories (adult, gambling, piracy).
    4. Resolves hostname to verify it doesn't point to private LAN / cloud metadata IPs.
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"

    if parsed.scheme not in ("http", "https"):
        return False, "Only HTTP and HTTPS URLs are permitted."

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "Invalid URL: missing hostname."

    # 1. Direct SSRF check on hostname
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or hostname.endswith(".local") or hostname.endswith(".internal"):
        return False, f"Access to internal hostname '{hostname}' is blocked (SSRF Protection)."

    # 2. Category Keyword Guard (Gambling, Adult, Unauthorized)
    if any(k in hostname for k in FORBIDDEN_CATEGORY_KEYWORDS):
        return False, f"Navigation to '{hostname}' is blocked by safety policy (restricted content category)."

    # 3. DNS-level SSRF resolution check (catches rebinding, 169.254.x.x, 192.168.x.x, etc.)
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            if _is_ip_private(ip_str):
                return False, f"Domain '{hostname}' resolves to private/internal IP {ip_str} (SSRF Protection)."
    except socket.gaierror:
        # Hostname doesn't resolve in DNS; browser navigation will fail naturally
        pass

    return True, ""


# --- Layer 2: Semantic AI Guard & Reputation Cache --------------------------
# Caches domain safety verdicts (safe: bool, reason: str, timestamp: float)
# Prevents repeating model calls for already approved sites.
_domain_reputation_cache: Dict[str, tuple] = {}
DOMAIN_CACHE_TTL_SECONDS = 3600  # 1 hour

TRUSTED_DOMAINS = {
    "google.com", "bing.com", "duckduckgo.com", "github.com",
    "wikipedia.org", "mozilla.org", "w3schools.com", "stackoverflow.com",
    "python.org", "npmjs.com", "pypi.org"
}


def is_domain_trusted(hostname: str) -> bool:
    h = hostname.lower()
    for td in TRUSTED_DOMAINS:
        if h == td or h.endswith("." + td):
            return True
    return False


async def check_semantic_guard(url: str, user_intent: str) -> tuple[bool, str]:
    """
    Production-Grade Layer 2 Semantic Guard:
    1. Checks trusted domain bypass (google, github, etc.).
    2. Checks in-memory domain reputation cache (0ms, 0 tokens).
    3. Evaluates destination safety & task relevance via a small, fast model.
    4. Enforces a strict 3.5s timeout with fail-closed security.
    """
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return False, "Malformed URL"

    if not hostname:
        return False, "Missing hostname"

    # Fast Path 1: Known safe infrastructure/tech domains
    if is_domain_trusted(hostname):
        return True, "Trusted domain"

    # Fast Path 2: In-memory reputation cache
    now = time.time()
    if hostname in _domain_reputation_cache:
        cached_safe, cached_reason, ts = _domain_reputation_cache[hostname]
        if now - ts < DOMAIN_CACHE_TTL_SECONDS:
            return cached_safe, cached_reason

    # Layer 2 Model Evaluation
    try:
        from graph.llm_factory import get_guard_model
        from langchain_core.messages import SystemMessage, HumanMessage

        guard_model = get_guard_model()
        system_msg = (
            "You are a strict Enterprise Content Safety and Security Guard for an autonomous browser agent. "
            "Your job is to determine whether navigating to a destination URL is safe, ethical, and appropriate for the user's task.\\n\\n"
            "STRICT PROHIBITIONS:\\n"
            "- Block adult/sexual/pornographic content.\\n"
            "- Block gambling, online casinos, betting, and lotteries.\\n"
            "- Block malware, piracy, torrents, phishing, or scam domains.\\n"
            "- Block domains that are completely irrelevant or harmful to enterprise use.\\n\\n"
            "Format your reply strictly as JSON: {\\\"safe\\\": true, \\\"reason\\\": \\\"ok\\\"} or {\\\"safe\\\": false, \\\"reason\\\": \\\"concise reason\\\"}."
        )

        user_content = (
            f"User Stated Goal/Task: \\\"{user_intent or 'General web browsing'}\\\"\\n"
            f"Target URL: \\\"{url}\\\"\\n"
            f"Target Domain: \\\"{hostname}\\\"\\n\\n"
            f"Is this destination safe and appropriate?"
        )

        async def _call_guard():
            resp = await guard_model.ainvoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=user_content)
            ])
            return resp.content

        raw_response = await asyncio.wait_for(_call_guard(), timeout=3.5)

        clean_text = raw_response.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(clean_text)
        is_safe = bool(parsed.get("safe", False))
        reason = str(parsed.get("reason", "Policy violation"))

        _domain_reputation_cache[hostname] = (is_safe, reason, now)
        return is_safe, reason

    except asyncio.TimeoutError:
        logger.warning(f"Semantic Guard timed out for '{hostname}'. Failing closed for security.")
        return False, "Security guard timeout. Verification could not complete."
    except Exception as e:
        logger.error(f"Semantic Guard error for '{hostname}': {e}")
        return False, f"Safety verification failed ({str(e)})"


def is_domain_blocked(url: str) -> bool:
    """True if url's host is explicitly blocked, or a subdomain of a
    blocked entry ('bank.com' blocked also blocks 'login.bank.com')."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    for domain in _load_blocked_domains():
        domain = domain.lower().strip()
        if domain and (host == domain or host.endswith("." + domain)):
            return True
    return False


# Clicks whose selector text suggests a payment, purchase, or other
# consequential action pause for human approval before they happen, through
# the SAME interrupt()/approve-or-reject flow already used for write-tool
# calls - this is the actual guardrail: gate the action, not the domain.
RISKY_CLICK_KEYWORDS = [
    "buy", "purchase", "pay", "checkout", "check out", "confirm order",
    "confirm payment", "place order", "subscribe", "send money", "transfer",
    "delete account", "remove account", "confirm purchase", "donate",
    "submit payment", "add to cart",
]


def _looks_risky(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in RISKY_CLICK_KEYWORDS)



def is_manual_control(session_id: str) -> bool:
    sess = _sessions.get(session_id)
    return bool(sess and sess.get("manual_control"))


def get_manual_control_info(session_id: str) -> dict:
    """What the frontend needs to render the 'you're in control' banner."""
    sess = _sessions.get(session_id)
    if not sess:
        return {"manual_control": False, "reason": "", "details": ""}
    return {
        "manual_control": bool(sess.get("manual_control")),
        "reason": sess.get("manual_control_reason", ""),
        "details": sess.get("manual_control_details", ""),
    }


async def enter_manual_control(session_id: str, reason: str, details: str = "") -> None:
    sess = _sessions.get(session_id)
    if sess:
        sess["manual_control"] = True
        sess["manual_control_reason"] = reason
        sess["manual_control_details"] = details
        logger.info(f"Session '{session_id}': handed control to the user ({reason}).")


async def exit_manual_control(session_id: str) -> None:
    """Called when the user clicks 'Resume agent' in the sandbox window."""
    sess = _sessions.get(session_id)
    if sess:
        sess["manual_control"] = False
        sess["manual_control_reason"] = ""
        sess["manual_control_details"] = ""
        logger.info(f"Session '{session_id}': user handed control back to the agent.")


async def dispatch_click(session_id: str, x: float, y: float) -> bool:
    """Replay a click the user made on the live sandbox image onto the real
    headless page, at the same (already viewport-scaled) coordinates."""
    page = await get_page(session_id)
    if not page:
        return False
    await page.mouse.click(x, y)
    log_audit("MANUAL_CLICK", f"Session: {session_id} | x={x:.0f}, y={y:.0f}")
    return True


async def dispatch_key(session_id: str, key: str) -> bool:
    """Replay a keystroke the user typed into the live sandbox onto the real
    page. `key` is expected to be a browser KeyboardEvent.key value ('a',
    'Enter', 'Backspace', ...), which lines up with Playwright's own key
    names for the vast majority of keys."""
    page = await get_page(session_id)
    if not page:
        return False
    try:
        await page.keyboard.press(key)
    except Exception:
        # Playwright didn't recognize the key by name - if it's a single
        # printable character, insert it literally instead of dropping it.
        if len(key) == 1:
            await page.keyboard.insert_text(key)
        else:
            return False
    # Never log what was actually typed - this is exactly where passwords
    # and OTPs get entered.
    log_audit("MANUAL_KEY", f"Session: {session_id} | user typed a key (content not logged)")
    return True


async def dispatch_scroll(session_id: str, delta_x: float, delta_y: float) -> bool:
    page = await get_page(session_id)
    if not page:
        return False
    await page.mouse.wheel(delta_x, delta_y)
    return True

def log_audit(action: str, details: str):
    """Write all important actions to the audit log."""
    timestamp = datetime.now().isoformat()
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] ACTION: {action} | DETAILS: {details}\n")
    logger.info(f"Audit Log: {action} - {details}")


async def take_screenshot(name: str, session_id: Optional[str] = None) -> str:
    """Take a screenshot for evidence retention."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{EVIDENCE_DIR}/{name}_{timestamp}.png"
    page = await get_page(session_id)
    if page:
        await page.screenshot(path=filename, full_page=False)
    return filename


async def get_latest_screenshot_base64(session_id: Optional[str] = None) -> Optional[str]:
    """Take a fast, low-quality screenshot and return as base64 string for the UI.
    Used both by the per-step chat screenshots (graph/mcp_graph.py) and by the
    continuous /ws/sandbox/{session_id} live-view feed in main.py."""
    page = await get_page(session_id)
    if page:
        try:
            image_bytes = await page.screenshot(type="jpeg", quality=40)
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None
    return None


async def _wait_for_spa_settle(page: Page, timeout_ms: int = 2500) -> None:
    """
    Step 4: SPA Settling & Hydration Helper.
    Waits for dynamic client-side hydration, async rendering, and network requests to settle:
    1. Awaits domcontentloaded.
    2. Awaits networkidle (capped at timeout_ms so long-polling ads/analytics never hang).
    3. Verifies body element is attached.
    4. Provides a brief 150ms tick to allow React/Vue/Next.js render loops to paint.
    """
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass

    try:
        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass

    try:
        await page.locator("body").wait_for(state="attached", timeout=timeout_ms)
    except Exception:
        pass

    await asyncio.sleep(0.15)


@tool
async def navigate(url: str) -> str:
    """
    Navigate to a specified URL.
    Use this tool to open websites. Browsing is open to any site by default;
    a short blocklist can explicitly rule a few domains out. Risky actions
    (payments, purchases, etc.) are approved separately when you try to
    click them, not restricted here.
    """
    sid = _current_session_id.get()
    if is_killed(sid):
        return "Error: The user pressed Emergency Stop. This task has been cancelled - stop here, do not retry or navigate anywhere, and tell the user their request was stopped."

    # 1. Security & Category validation (SSRF + Adult/Gambling guard)
    is_safe, block_reason = validate_navigation_security(url)
    if not is_safe:
        log_audit("SECURITY_BLOCKED", f"Session: {sid} | URL: {url} | Reason: {block_reason}")
        return f"Navigation Blocked: {block_reason} Do not retry this URL. Inform the user."

    # 2. Domain Blocklist check
    if is_domain_blocked(url):
        host = urlparse(url).hostname or url
        log_audit("DOMAIN_BLOCKED", f"Session: {sid} | URL: {url}")
        return (
            f"Error: '{host}' is explicitly blocked and cannot be visited. "
            f"Do not try another URL on this same site. Tell the user this domain is blocked."
        )

    # 3. Layer 2: Semantic AI Guardrail (Small Model check for task relevance & unclassified domains)
    user_task = _current_user_prompt.get()
    semantic_safe, semantic_reason = await check_semantic_guard(url, user_task)
    if not semantic_safe:
        log_audit("SEMANTIC_GUARD_BLOCKED", f"Session: {sid} | URL: {url} | Reason: {semantic_reason}")
        return f"Navigation Blocked: Safety policy violation - {semantic_reason}. Do not navigate to this site."

    page = await get_page(sid)
    if not page:
        page = await init_browser(sid)

    # Step 4: Resilient Navigation with 2-attempt retry & SPA hydration settling
    max_retries = 2
    last_error = None
    for attempt in range(1, max_retries + 1):
        log_audit("NAVIGATE", f"Session: {sid} | URL: {url} | Attempt: {attempt}")
        try:
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            await _wait_for_spa_settle(page, timeout_ms=2500)
            return f"Successfully navigated to {url}"
        except Exception as e:
            last_error = e
            logger.warning(f"Navigation to {url} (attempt {attempt}/{max_retries}) failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(1.0)
            else:
                return f"Failed to navigate to {url} after {max_retries} attempts: {str(last_error)}"


def _normalize_selector(sel: str) -> str:
    """
    Normalizes selectors so that if the LLM passes an ARIA snapshot entry
    (e.g. `button "Continue"`, `link "Sign up"`, `textbox "Email"`),
    it is automatically converted to a valid Playwright role selector
    `role=button[name="Continue"]`.
    """
    s = sel.strip()
    aria_match = re.match(
        r'^(button|link|textbox|checkbox|combobox|tab|heading|menuitem)\s+["\']([^"\']+)["\']',
        s,
        re.IGNORECASE,
    )
    if aria_match:
        role = aria_match.group(1).lower()
        name = aria_match.group(2)
        return f'role={role}[name="{name}"]'
    return s


async def _visualize_mouse_action(page: Page, locator) -> None:
    """Shows a visible glowing red cursor dot on the page where the agent clicks, so the live stream shows what the agent is doing."""
    try:
        box = await locator.bounding_box()
        if box:
            x = box["x"] + box["width"] / 2
            y = box["y"] + box["height"] / 2
            await page.evaluate(f"""(() => {{
                let cursor = document.getElementById('ai-agent-cursor');
                if (!cursor) {{
                    cursor = document.createElement('div');
                    cursor.id = 'ai-agent-cursor';
                    cursor.style.position = 'fixed';
                    cursor.style.width = '18px';
                    cursor.style.height = '18px';
                    cursor.style.borderRadius = '50%';
                    cursor.style.backgroundColor = 'rgba(239, 68, 68, 0.9)';
                    cursor.style.border = '2px solid white';
                    cursor.style.boxShadow = '0 0 12px rgba(239, 68, 68, 1)';
                    cursor.style.pointerEvents = 'none';
                    cursor.style.zIndex = '999999999';
                    cursor.style.transition = 'all 0.1s ease-out';
                    cursor.style.transform = 'translate(-50%, -50%)';
                    document.body.appendChild(cursor);
                }}
                cursor.style.left = '{x}px';
                cursor.style.top = '{y}px';
                cursor.style.display = 'block';
            }})()""")
    except Exception:
        pass


@tool
async def click(selector: str) -> str:
    """
    Click on an element in the browser.
    You can use:
    - Playwright role selectors: selector="role=button[name='Log In']" or selector="role=link[name='Home']"
    - Exact text selectors: selector="text='Log In'" or selector="text='Submit'"
    - ARIA snapshot format: selector='button "Log In"'
    Use this to click buttons, links, or open menus.
    A click that looks like a payment, purchase, subscription, transfer, or
    account deletion will pause and wait for the user's explicit approval
    before it happens - this is expected, not an error.
    """
    sid = _current_session_id.get()
    page = await get_page(sid)
    if not page:
        return "Error: Sandbox not initialized or was closed. Navigate to a page first."

    if _looks_risky(selector):
        log_audit("RISKY_CLICK_PENDING", f"Session: {sid} | Selector: {selector}")
        decision = interrupt({
            "type": "approval_request",
            "tool": "click",
            "args": {"selector": selector},
        })
        if decision != "approved":
            log_audit("RISKY_CLICK_REJECTED", f"Session: {sid} | Selector: {selector}")
            return f"Action blocked: clicking '{selector}' was not approved by the user. Do not retry it - tell the user the action was not approved."
        log_audit("RISKY_CLICK_APPROVED", f"Session: {sid} | Selector: {selector}")

    log_audit("CLICK", f"Session: {sid} | Selector: {selector}")
    await take_screenshot(f"pre_click_{selector.replace(' ', '_')}", sid)

    target_selector = _normalize_selector(selector)
    
    # Step 4: Resilient click with strict-mode protection, stale-element recovery, and SPA settle
    last_error = None
    for attempt in range(1, 3):
        try:
            # Use .first to avoid strict-mode crashes when duplicate buttons exist
            loc = page.locator(target_selector).first
            await _visualize_mouse_action(page, loc)
            await loc.click(timeout=4000)
            await _wait_for_spa_settle(page, timeout_ms=1500)
            await take_screenshot(f"post_click_{selector.replace(' ', '_')}", sid)
            return f"Successfully clicked on {target_selector}"
        except Exception as e:
            last_error = e
            if attempt == 1:
                await asyncio.sleep(0.3)
                # Fallback to raw selector if normalized was different
                if target_selector != selector:
                    try:
                        await page.locator(selector).first.click(timeout=3000)
                        await _wait_for_spa_settle(page, timeout_ms=1500)
                        await take_screenshot(f"post_click_{selector.replace(' ', '_')}", sid)
                        return f"Successfully clicked on {selector}"
                    except Exception:
                        pass
            else:
                return f"Failed to click on {selector}: {str(last_error)}"


@tool
async def type_text(selector: str, text: str) -> str:
    """
    Type text into an input field in the browser.
    You can use:
    - Playwright role selectors: selector="role=textbox[name='Email']"
    - Standard input selectors: selector="input[name='email']" or selector="input[placeholder='Search']"
    - ARIA snapshot format: selector='textbox "Email"'
    DO NOT use this tool for typing passwords! The user will take over for passwords.
    """
    sid = _current_session_id.get()
    page = await get_page(sid)
    if not page:
        return "Error: Sandbox not initialized or was closed. Navigate to a page first."

    # Security: Do not log the exact text to avoid leaking PII in audit
    log_audit("TYPE", f"Session: {sid} | Selector: {selector}, Text Length: {len(text)}")
    await take_screenshot(f"pre_type_{selector.replace(' ', '_')}", sid)

    target_selector = _normalize_selector(selector)
    
    # Step 4: Resilient input fill with strict-mode protection and retry
    last_error = None
    for attempt in range(1, 3):
        try:
            # Use .first to avoid strict-mode crash when multiple inputs match
            loc = page.locator(target_selector).first
            await _visualize_mouse_action(page, loc)
            await loc.fill(text, timeout=4000)
            await take_screenshot(f"post_type_{selector.replace(' ', '_')}", sid)
            return f"Successfully typed text into {target_selector}"
        except Exception as e:
            last_error = e
            if attempt == 1:
                await asyncio.sleep(0.3)
                if target_selector != selector:
                    try:
                        await page.locator(selector).first.fill(text, timeout=3000)
                        await take_screenshot(f"post_type_{selector.replace(' ', '_')}", sid)
                        return f"Successfully typed text into {selector}"
                    except Exception:
                        pass
            else:
                return f"Failed to type in {selector}: {str(last_error)}"


@tool
async def get_page_content() -> str:
    """
    Extract the semantic structure and text of the current page via Accessibility Tree (ARIA snapshot).
    Use this to read what is currently visible on the screen, identify buttons, links, inputs, and headings.
    Returns elements with their semantic roles and names, ready for role selectors or text selectors.
    """
    sid = _current_session_id.get()
    page = await get_page(sid)
    if not page:
        return "Error: Sandbox not initialized or was closed."

    log_audit("EXTRACT", f"Session: {sid} | Extracted page content")

    # Step 4: Ensure dynamic SPA has settled before taking snapshot
    await _wait_for_spa_settle(page, timeout_ms=1500)

    # Primary: Modern Playwright Accessibility Tree (ARIA Snapshot)
    try:
        snap = await page.locator("body").aria_snapshot()
        if snap and len(snap.strip()) > 0:
            clean_snap = re.sub(r"[\r\n]{3,}", "\n\n", snap).strip()
            if len(clean_snap) > 3500:
                clean_snap = clean_snap[:3500] + "\n...[accessibility tree truncated to save tokens]"
            return (
                "--- ACCESSIBILITY TREE (Interactive Elements & Structure) ---\n"
                f"{clean_snap}\n\n"
                "Selector Guide:\n"
                "- Click button: role=button[name=\"...\"] or text=\"...\"\n"
                "- Click link: role=link[name=\"...\"] or text=\"...\"\n"
                "- Fill input: role=textbox[name=\"...\"] or input[placeholder=\"...\"]"
            )
    except Exception as e:
        logger.warning(f"Accessibility snapshot failed, falling back to JS scraper: {e}")

    # Fallback: JavaScript scraper for raw text + top interactive elements
    try:
        js_script = """
        () => {
            // Target the main content area first to skip boilerplate headers/footers/navbars
            const mainEl = document.querySelector('main, article, [role="main"], #content, #main-content') || document.body;
            let text = (mainEl ? mainEl.innerText : document.body.innerText) || '';
            
            // Clean up excessive whitespace and blank lines
            text = text.replace(/[\\r\\n]{3,}/g, '\\n\\n').trim();
            if (text.length > 2500) {
                text = text.substring(0, 2500) + '\\n...[content truncated to save tokens]';
            }
            
            let interactives = '\\n\\n--- INTERACTIVE ELEMENTS (Top Visible) ---\\n';
            let count = 0;
            const elements = Array.from(document.querySelectorAll('input:not([type="hidden"]), button, select, textarea, a[href]'));
            
            for (const el of elements) {
                if (count >= 30) {
                    interactives += '...[additional elements omitted to save tokens]\\n';
                    break;
                }
                // Check if element is visible on screen
                const rect = el.getBoundingClientRect();
                const isVisible = rect.width > 0 && rect.height > 0 && 
                                  window.getComputedStyle(el).display !== 'none' && 
                                  window.getComputedStyle(el).visibility !== 'hidden';
                if (!isVisible) continue;
                
                let name = el.innerText || el.placeholder || el.getAttribute('aria-label') || el.name || el.value || '';
                name = name.trim().replace(/[\\r\\n]+/g, ' ').substring(0, 35);
                if (!name && el.tagName.toLowerCase() === 'a') continue; // Skip icon/unnamed links
                if (!name) name = 'unnamed';
                
                let type = el.tagName.toLowerCase();
                let sel = el.id ? `#${el.id}` : (el.name ? `[name='${el.name}']` : (el.placeholder ? `input[placeholder='${el.placeholder}']` : `text='${name}'`));
                interactives += `[${type}] "${name}" -> Use Selector: ${sel}\\n`;
                count++;
            }
            return text + interactives;
        }
        """
        content = await page.evaluate(js_script)
        return content[:3500]  # Capped at 3500 chars to avoid prompt token explosion
    except Exception as e:
        return f"Failed to extract content: {str(e)}"


@tool
async def request_human_action(reason: str, details: str = "") -> str:
    """
    Call this tool when you encounter a Login page, OTP, or CAPTCHA.
    Hands control of the live sandbox window over to the human: they see a
    banner over the live view, can click and type directly into the real
    page themselves (their input is forwarded into this session's sandbox,
    never through you), and click "Resume agent" when they're done. Waits
    up to 5 minutes for that.
    """
    sid = _current_session_id.get()
    page = await get_page(sid)
    if not page:
        return "Error: Sandbox closed. Navigate to a page first."

    log_audit("REQUEST_HUMAN_ACTION", f"Session: {sid} | Reason: {reason}")
    await enter_manual_control(sid, reason, details)

    waited_seconds = 0
    while is_manual_control(sid):
        await asyncio.sleep(1)
        waited_seconds += 1
        if waited_seconds >= 300:
            await exit_manual_control(sid)
            return "Error: Human took too long to respond (timeout after 5 minutes)."

    # Auto-scrape the new page content once control is handed back!
    content = await get_page_content.ainvoke({})
    return f"Human has completed the action and handed control back. Here is the new screen content:\n\n{content}"

@tool
async def press_key(key: str) -> str:
    """
    Press a keyboard key on the active page.
    Common keys: 'Enter', 'Escape', 'Tab', 'ArrowDown', 'ArrowUp', 'Backspace'.
    Use 'Enter' to submit forms/searches.
    Use 'Escape' to dismiss popups, dropdowns, calendars, and modal dialogs.
    """
    sid = _current_session_id.get()
    page = await get_page(sid)
    if not page:
        return "Error: Sandbox not initialized or was closed."

    log_audit("PRESS_KEY", f"Session: {sid} | Key: {key}")
    try:
        await page.keyboard.press(key)
        await _wait_for_spa_settle(page, timeout_ms=1000)
        return f"Successfully pressed key: {key}"
    except Exception as e:
        return f"Failed to press key {key}: {str(e)}"


# A list of tools to export
BROWSER_TOOLS = [navigate, click, type_text, press_key, get_page_content, request_human_action]
