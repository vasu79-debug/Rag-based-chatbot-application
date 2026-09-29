# Demo 5: Single-Step API-Enabled MCP Agent

A highly scalable, production-grade AI Agent that implements the **Model Context Protocol (MCP)** to autonomously execute tools (API calls, RAG search) while maintaining persistent database-backed session memory.

---

## 🌟 Key Features

1. **Model Context Protocol (MCP)**:
   - Complete decoupling of AI reasoning and tool execution.
   - External MCP Server (`mcp_server.py`) hosts tools like `get_weather` (Open-Meteo API) and `search_krify_knowledge` (Hybrid RAG).
   - Fast initialization via Persistent `stdio_client` and FastAPI Lifespan management.

2. **Autonomous Tool-Calling Agent**:
   - Zero hardcoded routing. The LLM acts autonomously as an agent, deciding when to chat normally and when to request tools via native Pydantic-validated function calling.

3. **Stateless UI & Stateful Database**:
   - The React frontend is completely stateless, sending only the user's `question` and a `session_id`.
   - The backend handles all multi-turn memory via SQLite (`SQLChatMessageHistory`), representing a true enterprise microservice architecture.

4. **Live SSE Streaming**:
   - Real-time insight into the Agent's reasoning loop.
   - Asynchronous Python generators (`astream`) pipe live tool-calling stages (e.g., *"Running tool: get_weather..."*) directly to the frontend UI via Server-Sent Events (SSE).

5. **Configurable Persona**:
   - Fully customizable AI name, role, and tone driven entirely by the `config.py` / `.env` variables.

---

## 🚀 Quick Start

```bash
# In demo5-single-step-api/
./start.sh
```

- **Frontend UI:** [http://localhost:5177](http://localhost:5177)
- **FastAPI Backend:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Database Location:** `backend/data/chat_history.db`

---

## 📂 Project Structure

```
demo5-single-step-api/
├── start.sh                  # Single-command launcher for backend + frontend
├── backend/
│   ├── main.py               # FastAPI application (SSE, DB hooks, Lifespan)
│   ├── mcp_server.py         # The MCP Server hosting tools (Weather, RAG)
│   ├── config.py             # Persona configuration
│   ├── data/
│   │   └── chat_history.db   # SQLite DB storing persistent conversation threads
│   └── graph/
│       ├── mcp_graph.py      # The MCP Client & Agent Execution Loop
│       └── llm_factory.py    # LLM Initialization
└── frontend/
    └── src/
        ├── App.jsx           # Stateless React UI generating session_id
        ├── api.js            # Fetch calls for SSE streaming
        └── components/
            └── ChatView.jsx  # UI displaying live AI reasoning steps
```
