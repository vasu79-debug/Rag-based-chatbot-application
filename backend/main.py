"""
FastAPI Backend Application for Demo 4 (Hybrid Knowledge).
Provides endpoints for Hybrid Chat, Admin Document Upload, Index Management, and Health Checks.
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
from graph.workflow import hybrid_graph_app


app = FastAPI(
    title="Demo 4 - Hybrid General + Organisational Knowledge AI",
    description="Enterprise AI backend combining Private Document RAG (ChromaDB + BM25 + FlashRank) and General LLM Intelligence using LangGraph.",
    version="1.0.0",
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
    history: Optional[List[MessageTurn]] = Field(default=[], description="Past conversation history")


class CitationItem(BaseModel):
    source: str
    page: int
    format: str
    snippet: str
    score: float


class KnowledgeSection(BaseModel):
    label: str
    content: str
    citations: Optional[List[CitationItem]] = None
    has_citations: Optional[bool] = False


class ChatResponse(BaseModel):
    route: str
    content: Optional[str] = None
    answer: Optional[str] = None
    org_section: Optional[KnowledgeSection] = None
    general_section: Optional[KnowledgeSection] = None
    citations: List[CitationItem] = []
    citations_count: int = 0


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
    """Returns system status, active models, and vector database stats."""
    docs = document_service.list_documents()
    total_chunks = sum(d.get("chunks_count", 0) for d in docs)
    return {
        "status": "healthy",
        "provider": settings.AI_PROVIDER,
        "default_model": settings.AI_DEFAULT_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "indexed_documents_count": len(docs),
        "total_indexed_chunks": total_chunks,
    }


import json
import asyncio
from fastapi.responses import StreamingResponse

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat Interface:
    Routes the query through the LangGraph StateGraph (Router -> RAG & General Nodes -> Synthesizer)
    and returns a structured dual-knowledge response.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        # Convert history format
        formatted_history = [{"role": h.role, "content": h.content} for h in request.history]

        # Execute LangGraph Workflow
        state_input = {
            "question": request.question.strip(),
            "history": formatted_history,
        }

        result = hybrid_graph_app.invoke(state_input)
        final_out = result.get("final_output", {})

        return ChatResponse(
            route=final_out.get("route", "HYBRID"),
            content=final_out.get("content") or final_out.get("answer"),
            answer=final_out.get("answer") or final_out.get("content"),
            org_section=final_out.get("org_section"),
            general_section=final_out.get("general_section"),
            citations=final_out.get("citations", []),
            citations_count=final_out.get("citations_count", 0),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the request: {str(e)}",
        )


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Real-Time SSE Streaming Chat Interface:
    Streams ChatGPT-style inline search & thinking progress and delivers the synthesized response.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    q = request.question.strip()
    formatted_history = [{"role": h.role, "content": h.content} for h in request.history]
    state_input = {
        "question": q,
        "history": formatted_history,
    }

    async def event_generator():
        try:
            # 1. Initial ChatGPT-style search status
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'retrieving', 'label': f'Searching Krify knowledge base for \"{q}\"'})}\n\n"
            await asyncio.sleep(0.01)

            final_out = None
            for event in hybrid_graph_app.stream(state_input, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "router":
                        route = node_output.get("route", "HYBRID")
                        rag_sub = node_output.get("sub_questions", {}).get("rag") or q
                        if route in ("RAG", "HYBRID"):
                            yield f"data: {json.dumps({'type': 'stage', 'stage': 'retrieving', 'label': f'Searching Krify documents for \"{rag_sub}\"'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'stage', 'stage': 'reasoning', 'label': 'Thinking...'})}\n\n"
                    elif node_name == "rag_node":
                        citations = node_output.get("citations", [])
                        count = len(citations)
                        if count > 0:
                            yield f"data: {json.dumps({'type': 'stage', 'stage': 'retrieved', 'label': f'Found {count} relevant passages in Krify documents · Synthesizing...'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'stage', 'stage': 'reasoning', 'label': 'Thinking...'})}\n\n"
                    elif node_name == "general_node":
                        yield f"data: {json.dumps({'type': 'stage', 'stage': 'reasoning', 'label': 'Thinking...'})}\n\n"
                    elif node_name == "synthesizer":
                        final_out = node_output.get("final_output", {})

                await asyncio.sleep(0.01)

            if not final_out:
                final_out = {
                    "route": "HYBRID",
                    "org_section": None,
                    "general_section": {"label": "Response", "content": "Completed."}
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
