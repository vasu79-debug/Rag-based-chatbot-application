"""
FastAPI Backend Application for Demo 5 (Single Step MCP Tool Calling).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from rag.document_service import document_service
from graph.mcp_graph import mcp_graph_app
from langchain_core.messages import HumanMessage, AIMessage

from contextlib import asynccontextmanager, AsyncExitStack
from mcp.client.sse import sse_client
from mcp import ClientSession
import sys
import os

mcp_state = {
    "url": "http://127.0.0.1:8001/sse",
    "exit_stack": None,
    "session": None
}

async def connect_mcp(url: str):
    if mcp_state.get("exit_stack"):
        await mcp_state["exit_stack"].aclose()
        
    stack = AsyncExitStack()
    try:
        read, write = await stack.enter_async_context(sse_client(url))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        mcp_graph_app.mcp_session = session
        mcp_state["exit_stack"] = stack
        mcp_state["session"] = session
        mcp_state["url"] = url
    except Exception as e:
        await stack.aclose()
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await connect_mcp(mcp_state["url"])
    except Exception as e:
        print(f"Warning: Could not connect to default MCP server: {e}")
    yield
    if mcp_state.get("exit_stack"):
        await mcp_state["exit_stack"].aclose()

app = FastAPI(
    title="Demo 5 - Single-Step API-Enabled Chatbot",
    description="Enterprise AI backend demonstrating tool/function calling with Pydantic validation.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Schemas ---

class MessageTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question or request")
    session_id: str = Field(default="default_session", description="The session or thread ID for database memory")
    history: Optional[List[MessageTurn]] = Field(default=[], description="Deprecated frontend history")


class CitationItem(BaseModel):
    source: str
    page: int
    format: str
    snippet: str
    score: float





class KnowledgeSection(BaseModel):
    label: str
    content: str

class ChatResponse(BaseModel):
    route: str = "API"
    answer: str
    general_section: KnowledgeSection

class DocumentItem(BaseModel):
    doc_id: str
    filename: str
    format: str
    chunks_count: int
    pages_count: int
    status: Optional[str] = "indexed"

# --- REST API Endpoints ---

@app.get("/api/health")
async def health_check():
    """Returns system status."""
    return {
        "status": "healthy",
        "provider": settings.AI_PROVIDER,
        "default_model": settings.AI_DEFAULT_MODEL,
        "indexed_documents_count": 0,
        "total_indexed_chunks": 0,
    }

@app.get("/api/settings/mcp")
async def get_mcp_url():
    return {"url": mcp_state.get("url", "")}

@app.post("/api/settings/mcp")
async def update_mcp_url(payload: dict):
    url = payload.get("url")
    if not url:
        raise HTTPException(400, "URL is required")
    try:
        await connect_mcp(url)
        return {"status": "success", "url": url}
    except Exception as e:
        raise HTTPException(500, f"Failed to connect to MCP Server at {url}: {str(e)}")

import json
import asyncio
from fastapi.responses import StreamingResponse

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat Interface:
    Routes the query through the LangGraph tool-calling graph.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        # Execute Workflow with DB Memory
        state_input = {
            "question": request.question.strip(),
            "session_id": request.session_id,
            "route": "api"
        }

        result = await mcp_graph_app.ainvoke(state_input)
        final_message = result["final_message"]

        return ChatResponse(
            route="API",
            answer=final_message,
            general_section=KnowledgeSection(
                label="AI Agent Response",
                content=final_message
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the request: {str(e)}",
        )

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Real-Time SSE Streaming Chat Interface.
    (Simplified for API tool calling - just returns the final result at once)
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    q = request.question.strip()
    session_id = request.session_id
    
    async def event_generator():
        try:
            state_input = {
                "question": q,
                "session_id": session_id,
                "route": "api"
            }

            async for event in mcp_graph_app.astream(state_input):
                if event["type"] == "stage":
                    # Send live tool/reasoning updates
                    yield f"data: {json.dumps({'type': 'stage', 'stage': 'reasoning', 'label': event['label']})}\n\n"
                    await asyncio.sleep(0.01)
                
                elif event["type"] == "complete":
                    final_message = event["final_message"]
                    final_out = {
                        "route": "API",
                        "answer": final_message,
                        "general_section": {"label": "AI Agent Response", "content": final_message}
                    }
                    yield f"data: {json.dumps({'type': 'complete', 'result': final_out})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )





@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Retrieve chat history for a given session."""
    history = mcp_graph_app.get_history(session_id)
    formatted_messages = []
    for msg in history.messages:
        if msg.type == "human":
            formatted_messages.append({"role": "user", "content": msg.content})
        elif msg.type == "ai" and msg.content:
            formatted_messages.append({
                "role": "assistant",
                "payload": {
                    "answer": msg.content,
                    "general_section": {"label": "AI Agent Response", "content": msg.content}
                }
            })
    return {"messages": formatted_messages}

@app.get("/api/admin/documents", response_model=List[DocumentItem])
async def list_documents():
    """Lists all documents currently indexed in ChromaDB."""
    docs = document_service.list_documents()
    return docs


@app.get("/api/admin/chunks")
async def list_all_stored_chunks(doc_id: Optional[str] = None):
    """
    Inspects raw chunk text, metadata, and chunk IDs stored inside ChromaDB.
    Optional query parameter `doc_id` to filter chunks for a specific document.
    """
    all_chunks = document_service.vector_store.get_all_chunks()
    
    if doc_id:
        all_chunks = [c for c in all_chunks if c.metadata.get("doc_id") == doc_id]

    chunks_data = []
    for i, c in enumerate(all_chunks, 1):
        chunks_data.append({
            "index": i,
            "chunk_id": c.metadata.get("chunk_id", f"chunk_{i}"),
            "doc_id": c.metadata.get("doc_id", "unknown"),
            "source": c.metadata.get("source", "unknown"),
            "page": c.metadata.get("page", 1),
            "format": c.metadata.get("format", "text"),
            "char_length": len(c.page_content),
            "content": c.page_content,
        })

    return {
        "total_chunks": len(chunks_data),
        "chunks": chunks_data,
    }



@app.post("/api/admin/documents", response_model=DocumentItem, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Admin Upload Endpoint:
    Accepts PDF, DOCX, CSV, or TXT files, extracts text, chunks, embeds into ChromaDB,
    and updates the BM25 index.
    """
    allowed_extensions = {".pdf", ".docx", ".doc", ".csv", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(allowed_extensions)}",
        )

    # Save to temporary file for parsing
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = document_service.ingest_file(tmp_path, file.filename)
        return DocumentItem(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to ingest document '{file.filename}': {str(e)}",
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=4, description="Public HTTP/HTTPS URL of webpage to index")


@app.post("/api/admin/url", response_model=DocumentItem, status_code=status.HTTP_201_CREATED)
async def ingest_url_endpoint(request: UrlIngestRequest):
    """
    Admin URL Ingest Endpoint:
    Safely fetches webpage with SSRF & malware protection, strips harmful scripts/payloads,
    chunks, embeds into ChromaDB, and syncs BM25 index.
    """
    if not request.url.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL cannot be empty.",
        )

    try:
        result = document_service.ingest_url(request.url.strip())
        return DocumentItem(**result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index website content: {str(e)}",
        )



@app.delete("/api/admin/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Admin Delete Endpoint:
    Purges all vector embeddings for the specified doc_id and re-indexes BM25.
    """
    success = document_service.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' was not found.",
        )
    return {"status": "deleted", "doc_id": doc_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
