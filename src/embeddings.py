"""Create CPU-compatible sentence embeddings."""

import os

os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformer:
    """Load the sentence transformer on CPU."""
    return SentenceTransformer(model_name, device="cpu")


def embed_texts(texts: list[str], model: SentenceTransformer):
    """Convert text into normalized vectors for cosine-similarity search."""
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
