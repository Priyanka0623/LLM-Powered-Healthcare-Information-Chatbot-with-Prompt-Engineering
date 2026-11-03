"""Create and search a small in-memory FAISS index."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings import embed_texts


def create_faiss_index(
    texts: list[str],
    embedding_model: SentenceTransformer,
) -> faiss.Index:
    """Build a cosine-similarity index using normalized vectors."""
    if not texts:
        raise ValueError("At least one text chunk is needed to build the index.")

    vectors = np.asarray(embed_texts(texts, embedding_model), dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search_faiss(
    index: faiss.Index,
    query: str,
    embedding_model: SentenceTransformer,
    result_count: int,
) -> list[tuple[int, float]]:
    """Return chunk positions and similarity scores for a question."""
    if not query.strip() or result_count <= 0:
        return []

    query_vector = np.asarray(embed_texts([query], embedding_model), dtype=np.float32)
    scores, positions = index.search(query_vector, min(result_count, index.ntotal))
    return [
        (int(position), float(score))
        for position, score in zip(positions[0], scores[0])
        if position >= 0
    ]
