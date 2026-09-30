# Demo 6: Multi-Intent Chatbot API (MCP-enabled)

A robust API-enabled chatbot that uses the **Model Context Protocol (MCP)** to autonomously parse, decompose, and execute complex user queries containing multiple, independent intents.

---

## 🌟 Key Features

1. **Multi-Intent Decomposition**:
   - Analyzes incoming messages and splits them into distinct, independent requests.
   - Automatically handles as many distinct intents as the user includes in their message.

2. **Parallel Execution**:
   - Executes each parsed intent asynchronously in parallel using Python's `asyncio.gather`.
   - Radically reduces total latency for complex user prompts (e.g. "What's the weather in Tokyo and what's Krify's return policy?").

3. **Partial Success & Fault Tolerance**:
   - Each intent is executed in an isolated branch.
   - If one intent fails (e.g. API timeout, bad arguments), the others succeed.
   - Admin-configurable behavior via `ALLOW_PARTIAL_SUCCESS`: decide whether to return the successful parts with error notes, or reject the entire prompt and ask the user to simplify.

4. **Model Context Protocol (MCP)**:
   - External MCP Server hosts tools like `get_weather` and `search_krify_knowledge`.
   - The LLM automatically maps individual intents to the appropriate tools using native Pydantic-validated function calling.

5. **Stateless UI & Stateful Database**:
   - Real-time conversation memory via SQLite (`SQLChatMessageHistory`).
   - Unified final response synthesized from all parallel branches is saved as a single conversational turn.

---

## 🚀 Quick Start

```bash
# In demo6-multi-intent/
./start.sh
```

- **Frontend UI:** [http://localhost:5177](http://localhost:5177)
- **FastAPI Backend:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Database Location:** `backend/data/chat_history.db`

---

## 📂 Project Structure

```
demo6-multi-intent/
├── start.sh                  # Single-command launcher for backend + frontend
├── backend/
│   ├── main.py               # FastAPI application (SSE, DB hooks, Lifespan)
│   ├── mcp_server.py         # The MCP Server hosting tools (Weather, RAG)
│   ├── config.py             # Configs (including MAX_INTENTS_PER_MESSAGE)
│   ├── data/
│   │   └── chat_history.db   # SQLite DB storing persistent conversation threads
│   └── graph/
│       ├── mcp_graph.py      # The Intent Decomposer, Parallel Execution & Synthesizer
│       └── llm_factory.py    # LLM Initialization
└── frontend/
    └── src/
        ├── App.jsx           
        ├── api.js            
        └── components/
            └── ChatView.jsx  
```
