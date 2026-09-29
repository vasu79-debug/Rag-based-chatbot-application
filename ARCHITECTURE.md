# System Architecture: Demo 6b - Multi-Step Workflow Chatbot

An end-to-end architecture specification for Demo 6b, demonstrating how an AI Agent leverages a deterministic LangGraph Workflow and the Model Context Protocol (MCP) to extract intent, fetch subscription data, analyze it in Python, and synthesize a final response.

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

    subgraph MCPClient ["LangGraph Workflow (mcp_graph.py)"]
        Graph["StateGraph Workflow"]
        SQLStore["SQLChatMessageHistory"]
        LLM["LLM (Groq/OpenAI)"]
        
        ExtractorNode["Node: Extractor"]
        FetchNode["Node: Fetch Data"]
        AnalyzeNode["Node: Analyze (Python)"]
        SynthNode["Node: Synthesize"]
        ChatNode["Node: Chat"]
    end

    subgraph MCPServer ["MCP Server (mcp_server.py)"]
        Tools["FastMCP Subzillo Tools"]
        GetSubs["get_subscriptions"]
    end

    UI --> Session
    Session -->|question + session_id| ChatEP
    Lifespan -->|Establishes persistent Stdio connection| MCPServer
    
    ChatEP --> Graph
    Graph <--> SQLStore
    SQLStore <--> DB
    
    Graph --> ExtractorNode
    ExtractorNode -->|Intent: analyze_cost| FetchNode
    ExtractorNode -->|Intent: chat| ChatNode
    FetchNode --> AnalyzeNode
    AnalyzeNode --> SynthNode
    
    ExtractorNode --> LLM
    SynthNode --> LLM
    ChatNode --> LLM
    
    FetchNode -->|Execute Tool| MCPServer
    MCPServer --> GetSubs
    
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
    participant Agent as LangGraph Workflow
    participant LLM as LLM API
    participant MCPSrv as MCP Server
    participant DB as SQLite Database

    User->>Frontend: Enters query ("How much am I spending?")
    Frontend->>FastAPI: POST /api/chat/stream
    
    FastAPI->>Agent: astream(question, session_id)
    
    Agent->>LLM: [Node: Extractor] Is this a cost analysis query?
    LLM-->>Agent: YES
    
    Agent-->>FastAPI: yield stage: "Executing fixed pipeline workflow..."
    FastAPI-->>Frontend: SSE data (Executing...)
    
    Agent->>MCPSrv: [Node: Fetch Data] Execute 'get_subscriptions' via MCP
    MCPSrv-->>Agent: JSON Data (Subscriptions)
    
    Agent->>Agent: [Node: Analyze] Python logic calculates monthly spend
    
    Agent->>LLM: [Node: Synthesize] Generate final markdown response
    LLM-->>Agent: Final formatted answer
    
    Agent->>DB: Save User Query and AI Response
    Agent-->>FastAPI: yield complete: Final Content
    FastAPI-->>Frontend: SSE data (Final Result)
    Frontend-->>User: Displays final formatted answer
```

---

## 🧩 Core Architectural Paradigm Shifts

### 1. Deterministic Multi-Step Workflow
Unlike agents that autonomously decide which tool to call in a loop, this architecture enforces a strict pipeline (`extractor` -> `fetch_data` -> `analyze` -> `synthesize`). This provides better reliability for known enterprise use cases where the steps to fulfill a request are well-defined.

### 2. Hybrid LLM + Code Analysis
The workflow intentionally separates data retrieval from analysis. Instead of giving raw JSON to the LLM to do math (which can hallucinate), the `analyze` node uses standard Python to calculate accurate monthly totals, and then passes the pre-calculated report to the `synthesize` node for formatting.

### 3. MCP Tool Integration in specific nodes
The MCP Client session is initialized at the FastAPI lifecycle level and passed to the LangGraph workflow. The `fetch_data` node directly accesses the MCP session to invoke tools without exposing them to the LLM's autonomous function calling, ensuring tools are only called when the pipeline dictates.
