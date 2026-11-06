import asyncio
import base64
import contextvars
import os
import logging
from datetime import datetime
from typing import Dict, Optional
from langchain_core.tools import tool
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Playwright

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

    browser = await pw.chromium.launch(headless=True)
    context = await browser.new_context(viewport={"width": 1280, "height": 800})
    page = await context.new_page()
    _sessions[session_id] = {
        "browser": browser,
        "context": context,
        "page": page,
        # Human-takeover state: while manual_control is True, the frontend
        # sandbox window forwards the user's own mouse/keyboard into this
        # page instead of the agent acting on it. See request_human_action().
        "manual_control": False,
        "manual_control_reason": "",
        "manual_control_details": "",
    }
    logger.info(f"Sandbox browser started for session '{session_id}' (headless).")
    return page


async def get_page(session_id: Optional[str] = None) -> Optional[Page]:
    """Return the live page for a session without creating one. Used by the
    tools (via the current-session contextvar) and by the live-view websocket."""
    sid = session_id or _current_session_id.get()
    sess = _sessions.get(sid)
    if sess and not sess["page"].is_closed():
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


@tool
async def navigate(url: str) -> str:
    """
    Navigate to a specified URL.
    Use this tool to open websites.
    """
    sid = _current_session_id.get()
    if is_killed(sid):
        return "Error: The user pressed Emergency Stop. This task has been cancelled - stop here, do not retry or navigate anywhere, and tell the user their request was stopped."
    page = await get_page(sid)
    if not page:
        page = await init_browser(sid)

    log_audit("NAVIGATE", f"Session: {sid} | URL: {url}")
    await page.goto(url)
    await page.wait_for_load_state("domcontentloaded")
    return f"Successfully navigated to {url}"


@tool
async def click(selector: str) -> str:
    """
    Click on an element in the browser.
    Since you only see the text of the page, you MUST use Playwright text selectors to click things!
    Example: selector="text='Log In'" or selector="text='Submit'"
    Use this to click buttons, links, or open menus.
    """
    sid = _current_session_id.get()
    page = await get_page(sid)
    if not page:
        return "Error: Sandbox not initialized or was closed. Navigate to a page first."

    log_audit("CLICK", f"Session: {sid} | Selector: {selector}")
    await take_screenshot(f"pre_click_{selector.replace(' ', '_')}", sid)

    try:
        await page.click(selector, timeout=5000)
        await take_screenshot(f"post_click_{selector.replace(' ', '_')}", sid)
        return f"Successfully clicked on {selector}"
    except Exception as e:
        return f"Failed to click on {selector}: {str(e)}"


@tool
async def type_text(selector: str, text: str) -> str:
    """
    Type text into an input field in the browser.
    You MUST use Playwright text selectors, name attributes, or placeholders.
    Example: selector="input[name='email']" or selector="input[placeholder='Search']"
    DO NOT use this tool for typing passwords! The user will take over for passwords.
    """
    sid = _current_session_id.get()
    page = await get_page(sid)
    if not page:
        return "Error: Sandbox not initialized or was closed. Navigate to a page first."

    # Security: Do not log the exact text to avoid leaking PII in audit
    log_audit("TYPE", f"Session: {sid} | Selector: {selector}, Text Length: {len(text)}")
    await take_screenshot(f"pre_type_{selector.replace(' ', '_')}", sid)

    try:
        await page.fill(selector, text, timeout=5000)
        await take_screenshot(f"post_type_{selector.replace(' ', '_')}", sid)
        return f"Successfully typed text into {selector}"
    except Exception as e:
        return f"Failed to type in {selector}: {str(e)}"


@tool
async def get_page_content() -> str:
    """
    Extract the text content of the current page.
    Use this to read what is currently visible on the screen.
    """
    sid = _current_session_id.get()
    page = await get_page(sid)
    if not page:
        return "Error: Sandbox not initialized or was closed."

    log_audit("EXTRACT", f"Session: {sid} | Extracted page content")
    try:
        js_script = """
        () => {
            let text = document.body.innerText;
            let interactives = '\\n\\n--- INTERACTIVE ELEMENTS ---\\n';
            document.querySelectorAll('input, button, a').forEach(el => {
                let name = el.innerText || el.placeholder || el.name || el.value || 'unnamed';
                name = name.trim().replace(/\\n/g, ' ').substring(0, 40);
                if(name) {
                    let type = el.tagName.toLowerCase();
                    let sel = el.id ? `#${el.id}` : (el.name ? `[name='${el.name}']` : `text='${name}'`);
                    interactives += `[${type}] "${name}" -> Use Selector: ${sel}\\n`;
                }
            });
            return text + interactives;
        }
        """
        content = await page.evaluate(js_script)
        return content[:10000]  # Limit size to avoid context overflow
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

# A list of tools to export
BROWSER_TOOLS = [navigate, click, type_text, get_page_content, request_human_action]
