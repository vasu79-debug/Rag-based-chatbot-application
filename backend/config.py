"""
Configuration management for Demo 4 (Hybrid Knowledge).
Uses Pydantic Settings for strongly-typed environment variables and model setup.
"""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:5174,http://localhost:5177,http://127.0.0.1:5173,http://127.0.0.1:5177"

    # AI Model Provider
    AI_PROVIDER: str = "groq"  # groq, openai, openrouter, ollama, custom_openai
    AI_DEFAULT_MODEL: str = "openai/gpt-oss-20b"
    AI_TEMPERATURE: float = 0.2

    # API Keys
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None

    # Custom OpenAI-Compatible Endpoint (e.g. Kaggle/Cloudflare Ollama tunnel)
    CUSTOM_OPENAI_BASE_URL: Optional[str] = None
    CUSTOM_OPENAI_API_KEY: str = "dummy"  # most self-hosted servers accept any key

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    UPLOAD_DIR: str = "./data/uploads"

    # Embedding & Reranker Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-6-v2"

    # Hybrid Retrieval Tuning
    RAG_TOP_K_VECTOR: int = 20
    RAG_TOP_K_BM25: int = 20
    RAG_TOP_N_RERANK: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.25

    # Multi-Turn Memory Window (Number of past message turns to retain)
    MAX_HISTORY_TURNS: int = 10

    # ==========================================
    # Assistant Persona & Behavior Configuration
    # ==========================================
    ASSISTANT_NAME: str = "Krify AI Assistant"
    COMPANY_NAME: str = "Krify Software Technologies"
    ASSISTANT_ROLE_DESCRIPTION: str = "an enterprise AI assistant dedicated to helping with company services, software solutions, documents, and workplace operations"
    ASSISTANT_TONE: str = "professional, executive, clear, and helpful"
    CUSTOM_SYSTEM_INSTRUCTIONS: str = "Provide clean, well-structured answers using bullet points and formatting where helpful."
    OUT_OF_SCOPE_MESSAGE: str = "I am an enterprise AI assistant dedicated to helping with company knowledge, documents, services, and workplace operations. I cannot answer unrelated outside topics like entertainment, sports, or general trivia."

    # ASSISTANT_NAME: str = "Krify Sales Specialist"
    # COMPANY_NAME: str = "Krify Software Technologies"
    # ASSISTANT_ROLE_DESCRIPTION: str = "a proactive, persuasive B2B sales and solution consultant helping prospective clients understand our services and book project consultations"
    # ASSISTANT_TONE: str = "persuasive, professional, energetic, and solution-oriented"
    # CUSTOM_SYSTEM_INSTRUCTIONS: str = "Highlight Krify's key value propositions, emphasize client ROI and past successes, and always encourage the user to schedule a discovery call or request a project quotation."


    # ASSISTANT_NAME: str = "Krify Customer Support"
    # COMPANY_NAME: str = "Krify Software Technologies"
    # ASSISTANT_ROLE_DESCRIPTION: str = "a patient, empathetic customer care specialist resolving client questions, troubleshooting issues, and providing clear step-by-step guidance"
    # ASSISTANT_TONE: str = "empathetic, polite, patient, and reassuring"
    # CUSTOM_SYSTEM_INSTRUCTIONS: str = "Focus on solving problems step-by-step. If an issue requires human escalation, guide the user to contact support@krify.com with their ticket details."
    

    # ASSISTANT_NAME: str = "Krify Marketing Strategist"
    # COMPANY_NAME: str = "Krify Software Technologies"
    # ASSISTANT_ROLE_DESCRIPTION: str = "a creative digital marketing and content strategist helping draft compelling social media posts, case study summaries, newsletters, and promotional copy"
    # ASSISTANT_TONE: str = "creative, engaging, modern, and punchy"
    # CUSTOM_SYSTEM_INSTRUCTIONS: str = "Use engaging hooks, bullet points, call-to-actions (CTAs), and modern industry phrasing to make content stand out."


    # ASSISTANT_NAME = "Krify Solution Architect"
    # COMPANY_NAME = "Krify Software Technologies"
    # ASSISTANT_ROLE_DESCRIPTION = "a senior technical consultant advising clients on software architectures, tech stacks, cloud scalability, and development best practices"
    # ASSISTANT_TONE = "technical, precise, objective, and authoritative"
    # CUSTOM_SYSTEM_INSTRUCTIONS = "Provide deep technical insights, recommend modern architectural patterns, and compare technologies objectively."



    # ASSISTANT_NAME = "Krify HR Generalist"
    # COMPANY_NAME = "Krify Software Technologies"
    # ASSISTANT_ROLE_DESCRIPTION = "an internal HR assistant answering employee queries about policies, benefits, leave, holidays, and company procedures"
    # ASSISTANT_TONE = "friendly, clear, professional, and helpful"
    # CUSTOM_SYSTEM_INSTRUCTIONS = "Help employees find policy information quickly. If something needs HR review, suggest they contact hr@krify.com."


    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def absolute_chroma_dir(self) -> Path:
        p = Path(self.CHROMA_PERSIST_DIR)
        if not p.is_absolute():
            p = self.BASE_DIR / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def absolute_upload_dir(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        if not p.is_absolute():
            p = self.BASE_DIR / p
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
