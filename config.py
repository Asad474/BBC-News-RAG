"""Configuration settings for BBC News RAG pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration parameters."""

    csv_path: str = "bbc_news.csv"
    persist_directory: str = str(Path(__file__).parent / "chroma_db")
    collection_name: str = "bbc_news"

    # Embedding and LLM settings
    embedding_model: str = "gemini-embedding-2-preview"
    llm_model: str = "gemini-3.6-flash"
    llm_temperature: float | None = None

    # Retrieval and Ingestion settings
    top_k: int = 3
    batch_size: int = 100
    max_retries: int = 5
    initial_retry_delay: float = 5.0
    max_retry_delay: float = 60.0

    @property
    def google_api_key(self) -> str:
        """Validate and return the Google API key from environment."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please add GOOGLE_API_KEY='your_api_key' "
                "to your .env file or set it as an environment variable."
            )
        return api_key
