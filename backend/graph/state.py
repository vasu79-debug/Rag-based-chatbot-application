"""
LangGraph State Schema for Demo 4.
Defines the shared state passed between workflow nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class Citation(TypedDict):
    source: str
    page: int
    format: str
    snippet: str
    score: float


class AgentState(TypedDict):
    # Inputs
    question: str
    history: List[Dict[str, str]]

    # Routing
    route: str  # "GENERAL", "RAG", "HYBRID"
    sub_questions: Dict[str, str]

    # RAG Branch State
    rag_chunks: List[Document]
    citations: List[Citation]
    rag_answer: Optional[str]
    rag_available: bool

    # General Branch State
    general_answer: Optional[str]

    # Final Fused Output
    final_output: Dict[str, Any]
