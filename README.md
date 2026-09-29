# Subscription Assistant (Demo 6C - Agentic Multi-Tool Chatbot)

This project demonstrates an **Autonomous Agentic Workflow** using LangGraph and the Model Context Protocol (MCP). Unlike rigid rule-based workflows (e.g., Demo 5), this agent autonomously parses goals, dynamically selects tools, evaluates responses, and performs multi-step reasoning.

## Features

- **Agentic ReAct Loop**: The AI doesn't follow a hardcoded script. Given a goal like "reduce my spending", it decides which tools to call, observes the results, and acts again.
- **MCP Tool Integration**: 
  - **Database Tools**: `get_subscriptions`, `create_subscription`, `delete_subscription` interact safely with a PostgreSQL database.
  - **Logic Tools**: `get_alternative_plans` and `get_usage_statistics`.
  - **Web Search Tool**: `search_public_subscription_data` uses DuckDuckGo (`ddgs`) to scrape real-time pricing off the live internet.
- **Persistent Chat History**: Session histories are stored locally using SQLite, and a ChatGPT-style sidebar allows users to revisit and delete old sessions.
- **Modern UI**: A responsive, premium Light Theme interface built with React.

## Getting Started

### 1. Prerequisites
- Python 3.12+
- Node.js & npm
- PostgreSQL running locally (Database name: `subscriptions`)

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install psycopg[binary] langchain-mcp-adapters ddgs
```

Ensure your `.env` file contains your LLM API keys (e.g., `GROQ_API_KEY`) and your database URL:
```
DATABASE_URL="postgresql://postgres:password@localhost:5432/subscriptions"
```

Start the backend:
```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to interact with your Agent!
