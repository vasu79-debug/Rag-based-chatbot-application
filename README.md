# Demo 6b: Multi-Step Workflow Chatbot (MCP-enabled)

A robust API-enabled chatbot that uses a **LangGraph StateGraph Workflow** combined with the **Model Context Protocol (MCP)** to intelligently route, fetch, analyze, and synthesize user requests about subscription data.

---

## 🌟 Key Features

1. **Deterministic Workflow Routing**:
   - Uses an LLM to extract the user's intent upfront (e.g., asking about subscriptions vs. general chat).
   - Routes the request through a fixed pipeline (`extractor` -> `fetch_data` -> `analyze` -> `synthesize`).

2. **Model Context Protocol (MCP)**:
   - External FastMCP Server hosts tools for managing Subscriptions (`get_subscriptions`, `create_subscription`, `update_subscription`, `delete_subscription`).
   - The workflow securely invokes the `get_subscriptions` tool over stdio when needed.

3. **Pure Python Analysis Node**:
   - Demonstrates hybrid architecture where data is fetched via MCP but analyzed using deterministic Python code (calculating monthly spend) rather than relying solely on the LLM.

4. **Stateless UI & Stateful Database**:
   - Real-time conversation memory via SQLite (`SQLChatMessageHistory`).

5. **Streaming Workflow Updates**:
   - LangGraph's `astream` yields state changes in real-time, sending SSE updates to the frontend as the agent moves through the pipeline.

---

## 🚀 Quick Start

```bash
# In demo6b-multi-step-workflow/
./start.sh
```

- **Frontend UI:** [http://localhost:5177](http://localhost:5177)
- **FastAPI Backend:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Database Location:** `backend/data/chat_history.db`

---

## 📂 Project Structure

```
demo6b-multi-step-workflow/
├── start.sh                  # Single-command launcher for backend + frontend
├── backend/
│   ├── main.py               # FastAPI application (SSE, DB hooks, Lifespan)
│   ├── mcp_server.py         # The MCP Server hosting Subzillo Subscription tools
│   ├── database.py           # DB models for Subscriptions
│   ├── data/
│   │   └── chat_history.db   # SQLite DB storing persistent conversation threads
│   └── graph/
│       ├── mcp_graph.py      # LangGraph Workflow (Extractor -> Fetch -> Analyze -> Synthesize)
│       └── llm_factory.py    # LLM Initialization
└── frontend/
    └── src/
        ├── App.jsx           
        ├── api.js            
        └── components/
            └── ChatView.jsx  
```
