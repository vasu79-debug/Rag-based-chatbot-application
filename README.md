# Demo 7A: API-Based Transactional Agent (Human-in-the-Loop)

This project demonstrates an advanced, production-grade **Transactional AI Agent** with **Human-in-the-Loop (HITL)** capabilities. It builds upon the decoupled Multi-Server MCP architecture and introduces strict safety mechanisms for executing sensitive operations (Write actions).

## Core Philosophy: Preview, Approve, Verify
Whenever this agent needs to modify real business data (e.g., adding a subscription, deleting an account, or creating a Jira ticket), it strictly adheres to a three-part discipline:
1. **Preview:** The LLM gathers required arguments and presents a formatted Markdown table to the user for visual confirmation.
2. **Approve:** The LangGraph execution pauses via `interrupt()`, pushing an `approval_needed` event to the React UI. A hard boundary prevents execution until the user clicks `[Approve]`.
3. **Verify:** After execution resumes, the agent checks the return values from the MCP tool and synthesizes a final confirmation message.

## Features
- **Dynamic Human-in-the-Loop (HITL):** Configurable `WRITE_TOOLS` list in `config.py`. Any MCP tool matching a name in this list will automatically trigger a UI breakpoint.
- **Stateful Graph Execution:** Uses LangGraph's Checkpointer (`MemorySaver`) to persist the exact state of the agent across HTTP boundaries during approval pauses.
- **Multi-Server MCP Orchestrator:** Connect an infinite number of external MCP tool servers via the Admin Panel. The agent inherits all tools dynamically.
- **Agentic Fallback:** If MCP servers disconnect, the agent gracefully degrades to Native RAG tools without crashing.

## Setup Instructions

### 1. Start the External MCP Server(s)
To provide the agent with database tools (like `delete_subscription`):
```bash
cd backend
python mcp_server.py
```
*(Runs on port 8001)*

### 2. Start the FastAPI Backend
```bash
cd backend
python main.py
```
*(Runs on port 8000)*

### 3. Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```

## How to Test HITL
1. Open the UI and connect the MCP Server in the Admin Panel (`http://127.0.0.1:8001/sse`).
2. Ask the Chatbot: `"Cancel my Netflix subscription"`
3. The chatbot will ask you to confirm. Say `"yes"`.
4. The backend Graph will pause, and the UI will show an Action Approval box.
5. Click **Approve** to resume the graph and execute the database deletion!
