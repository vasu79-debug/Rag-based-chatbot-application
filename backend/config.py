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
    AI_PROVIDER: str = "groq"  # groq, openai, openrouter, ollama
    AI_DEFAULT_MODEL: str = "openai/gpt-oss-20b"
    AI_TEMPERATURE: float = 0.2

    # API Keys
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None

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
    MAX_HISTORY_TURNS: int = 8

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
