# Demo 7B Architecture: Browser Agent with Transparency

Safe, observable web browsing with LangGraph + Playwright, built the way production browser agents work:
**observe → act by element number → verify → gate risky steps → contain the browser.**

## 1. Core data flow

1. **User request** in chat, e.g. "Find FabExpress Santhi Inn in Puducherry for 21–23 Nov 2026, 4 adults, 2 rooms".
2. **Main agent** (`graph/mcp_graph.py`, LangGraph ReAct). Laya prunes MCP tools that are clearly not needed;
   browser, search and knowledge tools are always available. Past turns are reduced to a 3-turn window.
3. **Browser loop** (`browser_tools.py`):
   - `navigate(url)` – URL policy (scheme, SSRF, category, blocklist, semantic guard) then open a pre-filled search URL.
   - `get_page_content()` – page text excerpt + a **numbered list of interactive elements** (`[12] link "FabExpress Santhi Inn"`).
   - `find_on_page("words")` – searches the **whole** page, so long result lists are never cut off.
   - `click(index)`, `type_text(index, text, press_enter)`, `select_option(index, option)`, `press_key`, `scroll`, `go_back`.
   - Every action reports **what changed** (navigated / content changed / new tab / no change). Repeating the same
     no-op twice is refused (loop detection). JS dialogs are auto-dismissed and reported.
4. **Approval gate** – clicks / Enter that book, reserve, pay, confirm, buy or submit a payment / personal-details form
   call LangGraph `interrupt()`; the chat shows an approval card. Rejected → the agent must not retry.
5. **Human takeover** – `request_human_action` for login, OTP, CAPTCHA, passwords, card details and guest details.
   The agent **cannot** type into password / card / CVV / OTP fields (refused in code).
6. **Live view** – `/ws/sandbox/{session_id}` streams JPEG frames of the session's page (~4 fps) to the chat UI.

## 2. Sandbox & containment

- One isolated browser context per chat session, connected over CDP to the Browserless container
  (`docker-compose.yml`, bound to `127.0.0.1:3000`, `TOKEN` required).
- If the container is unreachable, an **unsandboxed local Chromium is used only when `ENVIRONMENT=development`**
  (override with `ALLOW_LOCAL_BROWSER_FALLBACK`). Otherwise the tool fails.
- **Network guard**: every request from the page (navigations, XHR, images, iframes) is checked; private / loopback /
  link-local / metadata IPs are aborted. The final URL after redirects is re-checked and the page is closed if it is
  not allowed. Known limit: DNS is resolved by the backend, not inside the browser, so DNS-rebinding is not fully covered.
- **Prompt injection**: page content is returned inside `<untrusted_web_content>` and the system prompt forbids following
  instructions from web pages. MCP tools declaring `[operation: write]` always need approval.
- Audit log (`audit.log`, rotated at 5 MB, typed text and personal data never logged) and evidence screenshots
  (`evidence/`, deleted after `EVIDENCE_RETENTION_HOURS`).

## 3. Optional: `browser_use_task` (off by default)

`ENABLE_BROWSER_USE_TASK=true` adds a read-only research sub-agent built on browser-use 1.x. browser-use 1.x executes
model-written Python as the backend's OS user and drives its own browser connection (not shown in the live view, not
covered by the per-action gates). Enable it only when the backend runs in a locked-down container with no secrets on disk.

## 4. Not done yet

- Authentication / per-user session ownership for chat, WebSocket and sandbox endpoints.
- CDP screencast instead of screenshot polling for the live view.
- Postgres/Redis for checkpoints and session state (needed for more than one backend worker).
