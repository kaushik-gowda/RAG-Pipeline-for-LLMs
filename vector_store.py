# ============================================================
# vector_store.py – Embedding & FAISS vector index
# ============================================================
"""
This module handles:
  1. Encoding text chunks into dense vector embeddings using
     a Sentence Transformer model.
  2. Building a FAISS index for fast similarity search.
  3. Querying the index to retrieve the top-k most relevant
     chunks for a given question.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, TOP_K

# Load the embedding model once at module level
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


class VectorStore:
    """In-memory FAISS-backed vector store for text chunks."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[str] = []

    # ----------------------------------------------------------
    # Building the index
    # ----------------------------------------------------------
    def build_index(self, chunks: list[str]) -> None:
        """Encode *chunks* and add them to a new FAISS index.

        Parameters
        ----------
        chunks : list[str]
            The text chunks to index.
        """
        self.chunks = chunks
        embeddings = _embedding_model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print(
            f"[vector_store] Indexed {len(chunks)} chunks "
            f"(dim={dimension})."
        )

    # ----------------------------------------------------------
    # Querying
    # ----------------------------------------------------------
    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Return the *top_k* chunks most similar to *query*.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int
            How many results to return.

        Returns
        -------
        list[dict]
            Each dict has keys ``chunk`` (str) and ``distance`` (float).
        """
        if self.index is None or not self.chunks:
            raise RuntimeError(
                "Index is empty. Call build_index() first."
            )

        query_embedding = _embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype="float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "distance": float(dist),
                })
        return results
