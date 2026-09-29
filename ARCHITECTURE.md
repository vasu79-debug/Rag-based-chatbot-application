# System Architecture: Demo 5 - Single-Step API-Enabled MCP Agent

An end-to-end architecture specification for Demo 5, demonstrating how an AI Agent leverages the Model Context Protocol (MCP) to autonomously execute tools (live APIs and RAG) with a stateless frontend and a stateful SQLite database backend.

---

## 🏛️ High-Level System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend (React + Vite · Port 5177)"]
        UI["Stateless Chat Interface"]
        Session["Session ID Generator"]
        SSE["SSE Stream Handler (Live Stages)"]
    end

    subgraph API ["FastAPI Backend (Port 8000)"]
        Lifespan["FastAPI Lifespan (MCP Init)"]
        ChatEP["POST /api/chat/stream"]
        DB[(SQLite DB\ndata/chat_history.db)]
    end

    subgraph MCPClient ["MCP Client & Agent Engine"]
        Graph["Agent Loop (astream)"]
        SQLStore["SQLChatMessageHistory"]
        LLM["LLM (Groq/OpenAI)"]
    end

    subgraph MCPServer ["MCP Server (mcp_server.py)"]
        Tools["Exposed FastMCP Tools"]
        WeatherAPI["get_weather\n(Open-Meteo)"]
        RAGAPI["search_krify_knowledge\n(ChromaDB + BM25)"]
    end

    UI --> Session
    Session -->|question + session_id| ChatEP
    Lifespan -->|Establishes persistent Stdio connection| MCPServer
    
    ChatEP --> Graph
    Graph <--> SQLStore
    SQLStore <--> DB
    
    Graph -->|1. Request Action| LLM
    LLM -->|2. Function Call| Graph
    Graph -->|3. Trigger Tool| MCPServer
    MCPServer --> WeatherAPI
    MCPServer --> RAGAPI
    
    Graph -->|Yields Live Status| ChatEP
    ChatEP -->|SSE Updates| SSE
    SSE --> UI
```

---

## 🔄 End-to-End Query Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant Frontend as React Frontend
    participant FastAPI as FastAPI Server
    participant SQLite as Database
    participant Agent as MCP Client Loop (mcp_graph.py)
    participant LLM as LLM API
    participant MCPSrv as MCP Server Subprocess

    User->>Frontend: Enters query ("What is the weather in London?")
    Frontend->>FastAPI: POST /api/chat/stream (question, session_id)
    
    FastAPI->>Agent: astream(question, session_id)
    Agent->>SQLite: Fetch past messages for session_id
    
    Agent-->>FastAPI: yield stage: "Analyzing intent..."
    FastAPI-->>Frontend: SSE data (Analyzing intent)
    
    Agent->>LLM: Send Context + Available Tools
    LLM-->>Agent: JSON Tool Call (get_weather, args: London)
    
    Agent-->>FastAPI: yield stage: "Running tool: get_weather..."
    FastAPI-->>Frontend: SSE data (Running tool)
    
    Agent->>MCPSrv: Execute 'get_weather' via MCP Stdio adapter
    MCPSrv-->>Agent: Live JSON API Result (Temperature, Windspeed)
    
    Agent-->>FastAPI: yield stage: "Synthesizing final response..."
    FastAPI-->>Frontend: SSE data (Synthesizing)
    
    Agent->>LLM: Send Tool Result Context
    LLM-->>Agent: Natural Language Output
    
    Agent->>SQLite: Save User Query and AI Response
    Agent-->>FastAPI: yield complete: Final Content
    FastAPI-->>Frontend: SSE data (Final Result)
    Frontend-->>User: Displays final formatted answer
```

---

## 🧩 Core Architectural Paradigm Shifts (From Demo 4)

### 1. The MCP Decoupling
In previous architectures, the tools (like RAG and SQL) were tightly coupled into the LangGraph state execution pipeline. In Demo 5, **the AI logic is fully decoupled from the tool execution via the Model Context Protocol (MCP)**.
- `mcp_server.py` hosts the tools securely in an isolated environment.
- `mcp_graph.py` acts strictly as the **Client Brain**, reading the tools over standard IO, attaching them to the LLM via `bind_tools`, and telling the server when to execute them.

### 2. Stateless UI + Database Memory
Instead of the frontend managing a heavy `history` array, the frontend is completely stateless.
- The UI generates a lightweight `session_id`.
- The backend intercepts this ID and leverages `SQLChatMessageHistory` to inject the historical conversational thread from an SQLite Database into the LLM context.
- This mirrors an enterprise SaaS application.

### 3. Asynchronous Live Streaming (`astream`)
Instead of hiding the execution pipeline behind a monolithic blocking request, the Agent Loop acts as an asynchronous python generator (`yield`). 
As the agent analyzes intent, executes tools, and synthesizes data, those granular status milestones are pushed instantly to the user's screen using Server-Sent Events (SSE).
