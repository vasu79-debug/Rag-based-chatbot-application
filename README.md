# Demo 7B: Browser Agent

This project demonstrates an advanced **Browser AI Agent** using **Playwright**. It provides the agent with tools to navigate the web, click elements, type text, and read page content, while keeping a human in the loop for sensitive actions (like typing passwords).

## Core Philosophy: Transparency and Auditing
When this agent navigates the web on your behalf, it adheres to strict observability rules:
1. **Headful Browser:** The Playwright browser is launched in non-headless mode, so you can watch exactly what the agent is doing in real-time.
2. **Audit Logging:** Every action (Navigate, Click, Type, Extract) is logged to an `audit.log` file with timestamps.
3. **Evidence Collection:** Before and after every interaction (click or type), the agent automatically takes a screenshot and saves it to the `evidence/` directory.

## Features
- **Web Navigation & Interaction:** The agent can browse websites, click on links, and type into input fields.
- **Evidence Retention:** Screenshots are taken at every step to ensure a clear trail of what the agent saw and did.
- **Human-in-the-Loop (HITL) for Secrets:** The agent is explicitly instructed not to type passwords. Instead, it pauses and allows the human user to take over the browser window to enter sensitive credentials.

## Setup Instructions

### 1. Start the FastAPI Backend
```bash
cd backend
python main.py
```
*(Runs on port 8000)*

### 2. Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```

## How to Test
1. Open the UI.
2. Ask the Chatbot: `"Go to hacker news and read the top headline."`
3. The backend will launch a visible Chromium browser.
4. Watch as the agent navigates, clicks, and extracts information.
5. Check the `backend/evidence/` directory for screenshot trails and `backend/audit.log` for action logs!
