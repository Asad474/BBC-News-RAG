"""Main application entry point orchestrating ingestion, search, and RAG QA."""

import sys

from config import Config
from ingestion import NewsIngestor
from rag import NewsRAGService, NewsSearchService


class NewsApp:
    """Orchestrator for news ingestion and question answering."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.ingestor = NewsIngestor(self.config)
        self.search_service = NewsSearchService(self.config)
        self.rag_service = NewsRAGService(
            config=self.config, search_service=self.search_service
        )

    def run_ingestion(self, limit: int | None = None) -> int:
        """Trigger the data ingestion pipeline."""
        print("Starting BBC News ingestion pipeline...")
        return self.ingestor.ingest(limit=limit)

    def search(self, query: str, top_k: int = 3):
        """Search the indexed news articles."""
        print(f"\n--- Searching for: '{query}' (top {top_k}) ---")
        results = self.search_service.search_with_scores(query=query, k=top_k)
        for rank, (doc, score) in enumerate(results, start=1):
            title = doc.metadata.get("title", "No Title")
            published = doc.metadata.get("published_date", "N/A")
            print(f"[{rank}] (Score: {score:.4f}) {title} ({published})")
            print(f"    Link: {doc.metadata.get('link', 'N/A')}")
        return results

    def ask(self, question: str) -> str:
        """Ask a question through the RAG pipeline."""
        print(f"\nQuestion: {question}")
        answer = self.rag_service.ask(question)
        print(f"\nAnswer:\n{answer}")
        return answer


def main():
    """Main execution flow."""
    app = NewsApp()

    # Ingest data if ChromaDB is currently empty
    vector_store = app.ingestor.get_vector_store()
    current_count = vector_store._collection.count()
    if current_count == 0:
        print("Vector database is empty. Running initial ingestion...")
        # Optional: set limit=500 for fast initial dev testing, or None for full ingestion
        app.run_ingestion()
    else:
        print(f"Vector store already contains {current_count} documents.")

    # Demonstration: Semantic search
    sample_query = "What is the latest news in the world?"
    app.search(query=sample_query, top_k=3)

    # Demonstration: RAG Question-Answering
    app.ask(sample_query)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
