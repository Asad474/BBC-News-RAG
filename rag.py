"""Retrieval and generation (RAG) module for BBC News."""

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import Config


class NewsSearchService:
    """Handles semantic similarity search and retrieval over the vector store."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        _ = self.config.google_api_key  # Ensure API key is configured
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model
        )
        self.vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.config.persist_directory,
        )

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Perform semantic similarity search and return relevant documents."""
        top_k = k or self.config.top_k
        return self.vector_store.similarity_search(query=query, k=top_k)

    def search_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Perform similarity search returning documents with distance scores."""
        top_k = k or self.config.top_k
        return self.vector_store.similarity_search_with_score(query=query, k=top_k)

    def get_retriever(self, k: int | None = None):
        """Return a LangChain retriever interface for chaining."""
        top_k = k or self.config.top_k
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})


class NewsRAGService:
    """End-to-end Retrieval-Augmented Generation pipeline using Gemini."""

    def __init__(
        self,
        config: Config | None = None,
        search_service: NewsSearchService | None = None,
    ) -> None:
        self.config = config or Config()
        self.search_service = search_service or NewsSearchService(self.config)

        llm_kwargs: dict[str, Any] = {"model": self.config.llm_model}
        if self.config.llm_temperature is not None:
            llm_kwargs["temperature"] = self.config.llm_temperature
        self.llm = ChatGoogleGenerativeAI(**llm_kwargs)

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert summarizer and analyst. Using the context provided, "
                "answer the question accurately and concisely. "
                "Context: {context}\n\n"
                "Question: {question}\nAnswer:"
            ),
        )
        self.qa_chain = self._build_chain()

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """Format retrieved documents into a single formatted context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Construct the LangChain LCEL RAG execution chain."""
        retriever = self.search_service.get_retriever()
        return (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Query the RAG system and return the synthesized answer."""
        response = self.qa_chain.invoke(question)
        return response


if __name__ == "__main__":
    rag = NewsRAGService()
    test_q = "What is the latest news in the world?"
    print(f"Testing RAG with question: {test_q}")
    print(rag.ask(test_q))
