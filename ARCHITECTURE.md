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

Instead of making silent API calls, the Browser Agent physically controls a Chromium instance. 

Key architectural decisions:
- **Headful Mode:** `headless=False` ensures the human user can always see what the agent is doing. This prevents silent background manipulation.
- **Global State:** A single `_page` and `_browser` instance is maintained globally. This allows the agent to navigate across multiple steps without losing context or session data.

## 3. Human-in-the-Loop & Security Controls

Web agents pose a high security risk. This architecture mitigates risks through:

- **Visibility:** By forcing headful mode, the user can monitor the agent's progress.
- **Credential Safety:** The agent is instructed via system prompt NOT to type passwords. If a login is required, it can wait while the user manually types their password into the visible browser window.
- **Emergency Kill Switch:** `main.py` exposes a `POST /api/kill` endpoint. If the user notices the agent doing something unexpected, clicking "Stop" in the UI instantly tears down the Playwright browser context, stopping the agent dead in its tracks.
