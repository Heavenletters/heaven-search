"""Embedding model wrapper — lightweight, CPU-friendly."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# 384-dim, ~80MB, excellent quality-to-speed ratio
MODEL_NAME = "all-MiniLM-L6-v2"

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME, device="cpu")
    return _embedder


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts → (N, 384) float32 array."""
    model = get_embedder()
    # Normalize for cosine similarity via dot product
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query → (384,) float32 array."""
    return embed_texts([text])[0]
