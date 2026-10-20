"""Data ingestion and preprocessing module for BBC News dataset."""

import hashlib
import time
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import Config


class NewsIngestor:
    """Handles loading, cleaning, deduplication, and vector store ingestion."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._embeddings: GoogleGenerativeAIEmbeddings | None = None

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Lazily initialize Google Generative AI embeddings."""
        if self._embeddings is None:
            _ = self.config.google_api_key  # Validates key exists
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model
            )
        return self._embeddings

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load BBC news dataset and perform deduplication and feature extraction.

        Note: A document splitter is intentionally omitted here because each row
        is already an atomic semantic chunk (average ~160 characters).
        """
        print(f"Loading raw dataset from '{self.config.csv_path}'...")
        df = pd.read_csv(self.config.csv_path)
        df.columns = [
            "Title",
            "Published_Date",
            "Guid",
            "Article_Link",
            "Description",
        ]

        # Deduplicate and remove rows with missing essential fields
        initial_count = len(df)
        df.dropna(subset=["Description", "Title", "Article_Link"], inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop_duplicates(subset=["Article_Link"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(
            f"Deduplication complete: retained {len(df)} unique articles from {initial_count} rows."
        )

        # Preprocessing features: Extract tags and news IDs
        df["Tags"] = (
            df["Guid"].str.extract(r"news/([a-zA-Z-]+)-\d+")[0].fillna("general")
        )
        df["News_ID"] = (
            df["Guid"]
            .str.extract(r"-(\d+)$")[0]
            .fillna(pd.Series(df.index.astype(str)))
        )

        # Concatenate Title + Description for optimal semantic retrieval
        df["content"] = df["Title"] + "\n" + df["Description"]
        return df

    def prepare_documents(self, df: pd.DataFrame) -> tuple[list[Document], list[str]]:
        """Transform dataframe rows into LangChain Document objects with deterministic IDs."""
        documents: list[Document] = []
        ids: list[str] = []

        # itertuples is substantially faster than iterrows
        for row in df.itertuples(index=False):
            # Deterministic MD5 hash of article link prevents duplicate insertions
            doc_id = hashlib.md5(str(row.Article_Link).encode("utf-8")).hexdigest()
            metadata: dict[str, Any] = {
                "news_id": str(row.News_ID),
                "tags": str(row.Tags),
                "title": str(row.Title),
                "published_date": str(row.Published_Date),
                "link": str(row.Article_Link),
            }
            documents.append(Document(page_content=str(row.content), metadata=metadata))
            ids.append(doc_id)

        return documents, ids

    def get_vector_store(self) -> Chroma:
        """Connect to persistent ChromaDB collection."""
        return Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.config.persist_directory,
        )

    def ingest(self, limit: int | None = None) -> int:
        """Ingest documents into ChromaDB with idempotency, batching, and backoff."""
        df = self.load_and_preprocess()
        if limit is not None:
            df = df.head(limit)
            print(f"Limiting ingestion to {limit} documents for this run.")

        documents, ids = self.prepare_documents(df)
        vector_store = self.get_vector_store()

        # Resumable ingestion: check IDs already indexed in ChromaDB
        existing_count = vector_store._collection.count()
        existing_ids: set[str] = set()
        if existing_count > 0:
            existing_data = vector_store.get()
            existing_ids = set(existing_data["ids"])
            print(f"ChromaDB already contains {existing_count} indexed documents.")

        pending_docs = [
            doc
            for doc, doc_id in zip(documents, ids, strict=True)
            if doc_id not in existing_ids
        ]
        pending_ids = [doc_id for doc_id in ids if doc_id not in existing_ids]

        total_pending = len(pending_docs)
        if total_pending == 0:
            print("All documents are already indexed in ChromaDB. Skipping ingestion.")
            return 0

        print(
            f"Indexing {total_pending} documents into ChromaDB in batches of {self.config.batch_size}..."
        )

        for i in range(0, total_pending, self.config.batch_size):
            batch_docs = pending_docs[i : i + self.config.batch_size]
            batch_ids = pending_ids[i : i + self.config.batch_size]
            self._add_batch_with_retry(vector_store, batch_docs, batch_ids)
            current_done = min(i + len(batch_docs), total_pending)
            print(f"Indexed {current_done}/{total_pending} documents...")

        final_count = vector_store._collection.count()
        print(f"Ingestion complete! Total documents in vector store: {final_count}")
        return total_pending

    def _add_batch_with_retry(
        self,
        vector_store: Chroma,
        batch_docs: list[Document],
        batch_ids: list[str],
    ) -> None:
        """Insert a batch into ChromaDB with exponential backoff on 429 rate limits."""
        delay = self.config.initial_retry_delay
        for attempt in range(1, self.config.max_retries + 1):
            try:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                return
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if is_rate_limit and attempt < self.config.max_retries:
                    print(
                        f"Encountered rate limit (429). Backing off for {delay:.1f}s "
                        f"(attempt {attempt}/{self.config.max_retries})..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, self.config.max_retry_delay)
                else:
                    raise


def main() -> None:
    """Run the news ingestion pipeline."""
    ingestor = NewsIngestor()
    ingestor.ingest()


if __name__ == "__main__":
    main()
