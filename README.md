# Demo 4: Hybrid General + Organisational Knowledge AI

A standalone, enterprise-grade AI assistant that combines **Private Organisational Knowledge (RAG)** with **General World Intelligence (LLM Reasoning)** using **LangGraph** orchestration, **ChromaDB**, **BM25**, and **FlashRank**.

---

## 🌟 Key Features

1. **Dual-Knowledge Architecture**:
   - **🏢 Organisational Knowledge Stream**: Strictly grounded in your uploaded documents (PDF, DOCX, CSV, TXT) with exact source citations and page numbers.
   - **🌐 General Knowledge Stream**: World intelligence, industry best practices, comparisons, email drafting, and code generation.
2. **Enterprise Hybrid Retrieval Engine**:
   - **Dense Vectors (ChromaDB)** for semantic meaning.
   - **Sparse BM25 Search** for exact contract numbers, codes, and acronyms.
   - **Reciprocal Rank Fusion (RRF)** merging vector + keyword candidate pools.
   - **Neural Cross-Encoder (FlashRank)** for re-scoring candidates before LLM synthesis.
3. **LangGraph StateGraph Workflow**:
   - Intelligent Query Router classifies questions into `GENERAL`, `RAG`, or `HYBRID`.
   - Dual-branch execution with structured synthesis.
4. **Admin Knowledge Portal**:
   - Drag-and-drop document upload (PDF, DOCX, CSV, TXT).
   - Real-time vector indexing and one-click deletion.
5. **100% Standalone**:
   - Independent Python FastAPI backend on port `8000`.
   - Independent React (Vite) frontend on port `5177`.

---

## 🚀 Quick Start

```bash
# In demo4-hybrid-knowledge/
./start.sh
```

- **Frontend UI:** [http://localhost:5177](http://localhost:5177)
- **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 📂 Project Structure

```
demo4-hybrid-knowledge/
├── start.sh                  # Single-command launcher for backend + frontend
├── backend/
│   ├── .env                  # API keys and hyperparameters
│   ├── main.py               # FastAPI application
│   ├── run.sh                # Backend runner
│   ├── requirements.txt      # Python dependencies
│   ├── rag/
│   │   ├── loaders.py        # PDF, DOCX, CSV, TXT extractors
│   │   ├── chunker.py        # Semantic text chunker
│   │   ├── vector_store.py   # ChromaDB vector store
│   │   ├── keyword_search.py # BM25 search manager
│   │   ├── reranker.py       # FlashRank neural cross-encoder
│   │   ├── hybrid_retriever.py# RRF + Reranker pipeline
│   │   └── document_service.py# Admin document ingestion & deletion
│   └── graph/
│       ├── state.py          # LangGraph state schema
│       ├── llm_factory.py    # Multi-provider LLM factory (Groq, OpenAI)
│       ├── router.py         # Query classifier node
│       ├── rag_node.py       # Cited organizational knowledge node
│       ├── general_node.py   # General intelligence node
│       ├── synthesizer.py    # Dual stream merger node
│       └── workflow.py       # Compiled StateGraph
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx           # Main React component
        ├── api.js            # REST API client
        ├── index.css         # Dark glassmorphism design system
        └── components/
            ├── Header.jsx    # Navigation & health status
            ├── ChatView.jsx  # Dual-stream labeled chat UI
            ├── AdminPortal.jsx # Drag-and-drop document manager
            └── CitationsDrawer.jsx # Slide-over source passage viewer
```
