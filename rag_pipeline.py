# ============================================================
# rag_pipeline.py – Orchestrates the full RAG pipeline
# ============================================================
"""
High-level orchestrator that ties together the retriever,
vector store and generator into a single, easy-to-use class.
"""

from retriever import get_wikipedia_content, split_text
from vector_store import VectorStore
from generator import generate_answer
from config import TOP_K


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.topic: str | None = None
        self.document: str | None = None
        self.chunks: list[str] = []

    # ----------------------------------------------------------
    # Step 1 & 2: Ingest a topic
    # ----------------------------------------------------------
    def ingest(self, topic: str) -> dict:
        """Fetch Wikipedia content for *topic*, chunk it, and index it.

        Returns
        -------
        dict
            ``success`` (bool), ``message`` (str), ``num_chunks`` (int).
        """
        print(f"[pipeline] Ingesting topic: {topic}")

        document = get_wikipedia_content(topic)
        if document is None:
            return {
                "success": False,
                "message": f"Could not retrieve Wikipedia content for '{topic}'.",
                "num_chunks": 0,
            }

        self.topic = topic
        self.document = document
        self.chunks = split_text(document)

        print(f"[pipeline] Created {len(self.chunks)} chunks.")

        self.vector_store.build_index(self.chunks)

        return {
            "success": True,
            "message": f"Successfully indexed '{topic}' ({len(self.chunks)} chunks).",
            "num_chunks": len(self.chunks),
        }

    # ----------------------------------------------------------
    # Step 3 & 4: Ask a question
    # ----------------------------------------------------------
    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        """Retrieve relevant chunks and generate an answer.

        Parameters
        ----------
        question : str
            The user's question about the ingested topic.
        top_k : int
            Number of chunks to retrieve.

        Returns
        -------
        dict
            ``answer`` (str), ``score`` (float),
            ``retrieved_chunks`` (list[str]).
        """
        if not self.chunks:
            return {
                "answer": "",
                "score": 0.0,
                "retrieved_chunks": [],
                "error": "No topic has been ingested yet. Call ingest() first.",
            }

        # Retrieve
        results = self.vector_store.search(question, top_k=top_k)
        retrieved_chunks = [r["chunk"] for r in results]
        context = " ".join(retrieved_chunks)

        # Generate
        answer_data = generate_answer(question, context)

        return {
            "answer": answer_data["answer"],
            "score": answer_data["score"],
            "retrieved_chunks": retrieved_chunks,
        }
