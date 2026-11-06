# Demo 7B Architecture: Browser Agent with Transparency

This document outlines the architectural patterns used to achieve safe and observable web browsing using LangGraph and Playwright.

## 1. The Core Data Flow

When a user requests a web-based action (e.g., "Go to Hacker News and read the top story"), the system uses a ReAct agent to orchestrate a browser.

1. **User Request:** The user submits a natural language request.
2. **Tool Selection:** The LangGraph ReAct agent determines which browser action to take (e.g., `navigate(url)`).
3. **Browser Automation:** The agent calls a tool in `browser_tools.py`. Playwright executes the action in a **headful** (visible) browser window.
4. **Evidence Collection:** Before and after any interaction (`click` or `type_text`), Playwright takes a screenshot of the DOM state and saves it to the `evidence/` directory.
5. **Auditing:** Every action is securely logged to `audit.log`.
6. **Observation:** The result (or the extracted text) is returned to the agent, which then formulates the next step or the final answer.

## 2. Playwright Integration (`browser_tools.py`)

Instead of making silent API calls, the Browser Agent physically controls a Chromium instance - but it no longer does this as a visible window on the server's own desktop. It runs **headless, one isolated browser/context/page per chat session**, and streams what it sees into the chat UI itself as a live "sandbox window" - the same idea as ChatGPT's agent mode.

Key architectural decisions:
- **Headless + Per-Session Sandbox:** `_sessions: Dict[session_id, {browser, context, page}]` replaces the old single global `_page`/`_browser`. Each chat session gets its own isolated Chromium sandbox, so concurrent chats never share a browser, and the backend has no dependency on a display (it can run on a headless server/container).
- **Session binding via contextvar:** The LLM-facing tools (`navigate`, `click`, `type_text`, ...) take no `session_id` argument - the agent graph (`graph/mcp_graph.py`) calls `set_current_session(session_id)` once per turn before invoking the tools, and a `contextvars.ContextVar` carries that binding through to whichever sandbox the tool call should act on.
- **Live view over WebSocket:** `main.py` exposes `GET /ws/sandbox/{session_id}`, which polls `get_latest_screenshot_base64()` roughly every 250ms and pushes JPEG frames (plus the current page URL) to the frontend's `SandboxPanel.jsx`. This is a continuous live view, not just the one still-frame-per-tool-call that used to show up inline in the chat transcript.

## 3. Human-in-the-Loop & Security Controls

Web agents pose a high security risk. This architecture mitigates risks through:

- **Visibility:** The sandbox window shows the agent's live view continuously, so the user can monitor its progress even though the browser itself is headless.
- **Credential Safety:** The agent is instructed via system prompt NOT to type passwords.
- **Emergency Kill Switch:** `main.py` exposes `POST /api/kill?session_id=...`, used by both the header's "Emergency Stop" button and the sandbox panel's own "Stop" button. It tears down that session's sandbox browser context immediately.
- **Known gap - human takeover:** `request_human_action` still injects a "Resume" button into the page DOM and waits for it to be clicked, which only worked when the browser was a real window on the presenter's own screen. Now that the browser is headless and only visible as a screenshot stream, nobody can click that injected button - mouse/keyboard input forwarding from the sandbox panel into Playwright (`page.mouse` / `page.keyboard`, driven by events captured on the live-view `<img>`) is a deliberate follow-up, not yet built. Until it ships, login/OTP/CAPTCHA pauses will time out after 5 minutes rather than actually letting the user complete them in-sandbox.
