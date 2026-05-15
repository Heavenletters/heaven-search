"""Multi-backend embedding with auto-failover.

Backends:
    vertex  — Google text-embedding-004 via Generative Language API (768-dim, cloud)
    local   — BAAI/bge-base-en-v1.5 via sentence-transformers (768-dim, offline)
    minilm  — all-MiniLM-L6-v2 via sentence-transformers (384-dim, legacy offline)

Configure with EMBEDDING_BACKEND env var (default: vertex).
At query time, VertexEmbedder auto-falls-back to the local model on any error.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

import numpy as np


# ── Abstract interface ──────────────────────────────────────────────

class Embedder(ABC):
    """Abstract embedding backend."""

    dim: int

    @abstractmethod
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts → (N, dim) float32 array (L2-normalized)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query → (dim,) float32 array."""
        return self.embed_texts([text])[0]


# ── Vertex AI (cloud) ───────────────────────────────────────────────

class VertexEmbedder(Embedder):
    """Google text-embedding-004 via Generative Language API.

    Uses API key auth — no OAuth or service account needed.
    Batch endpoint supports up to 100 texts per request.
    """

    dim = 768

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("VERTEX_API_KEY", "")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004"
        self.max_batch = 100  # API limit per batchEmbedContents call

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import urllib.request
        import json

        if not self.api_key:
            raise RuntimeError("VERTEX_API_KEY not set")

        all_embeddings: list[np.ndarray] = []
        batch_size = min(batch_size, self.max_batch)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import urllib.request
        import json

        requests_payload = [
            {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": t}]},
            }
            for t in texts
        ]

        body = json.dumps({"requests": requests_payload}).encode("utf-8")
        url = f"{self.base_url}:batchEmbedContents?key={self.api_key}"

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Vertex API request failed: {e}")

        if "error" in data:
            raise RuntimeError(f"Vertex API error: {data['error']}")

        embeddings = []
        for entry in data.get("embeddings", []):
            values = entry.get("values", [])
            if not values:
                raise RuntimeError("Vertex API returned empty embedding")
            vec = np.array(values, dtype=np.float32)
            # L2-normalize for cosine similarity via dot product
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)


# ── Local sentence-transformers backends ────────────────────────────

class SentenceTransformerEmbedder(Embedder):
    """Base for local sentence-transformers models."""

    model_name: str
    _model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class BGEBaseEmbedder(SentenceTransformerEmbedder):
    """BAAI/bge-base-en-v1.5 — 768-dim, ~438MB, strong offline quality."""

    dim = 768
    model_name = "BAAI/bge-base-en-v1.5"


class MiniLMEmbedder(SentenceTransformerEmbedder):
    """all-MiniLM-L6-v2 — 384-dim, ~80MB, fast but shallow (legacy)."""

    dim = 384
    model_name = "all-MiniLM-L6-v2"


# ── Auto-failover embedder (for query time) ─────────────────────────

class AutoEmbedder(Embedder):
    """Try primary backend, fall back to secondary on any error.

    Primary is typically Vertex (cloud), secondary is bge-base (local).
    Both must have the same dimension for vector store compatibility.
    """

    def __init__(self, primary: Embedder, fallback: Embedder):
        if primary.dim != fallback.dim:
            raise ValueError(
                f"Primary dim ({primary.dim}) != fallback dim ({fallback.dim}). "
                "Both backends must use the same embedding dimension."
            )
        self.primary = primary
        self.fallback = fallback
        self.dim = primary.dim
        self._primary_healthy = True

    @property
    def primary_healthy(self) -> bool:
        return self._primary_healthy

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self._primary_healthy:
            try:
                return self.primary.embed_texts(texts, batch_size)
            except Exception as e:
                print(f"  ⚠ Primary embedder failed ({e}), falling back to local")
                self._primary_healthy = False

        return self.fallback.embed_texts(texts, batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        if self._primary_healthy:
            try:
                return self.primary.embed_query(text)
            except Exception:
                self._primary_healthy = False

        return self.fallback.embed_query(text)


# ── Factory / singleton ─────────────────────────────────────────────

def get_embedder(backend: str | None = None) -> Embedder:
    """Get an embedding backend by name.

    Args:
        backend: "vertex", "local", "minilm", or "auto".
                 Defaults to EMBEDDING_BACKEND env var, then "auto".

    Returns:
        Embedder instance. "auto" returns an AutoEmbedder wrapping
        vertex (primary) + local (fallback).
    """
    backend = backend or os.environ.get("EMBEDDING_BACKEND", "auto")

    if backend == "vertex":
        return VertexEmbedder()
    elif backend == "local":
        # Suppress HF hub warnings for offline use
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return BGEBaseEmbedder()
    elif backend == "minilm":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return MiniLMEmbedder()
    elif backend == "auto":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        vertex = VertexEmbedder()
        local = BGEBaseEmbedder()
        return AutoEmbedder(primary=vertex, fallback=local)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")


def get_dim(backend: str | None = None) -> int:
    """Get the embedding dimension for a given backend without loading the model."""
    dims = {"vertex": 768, "local": 768, "minilm": 384, "auto": 768}
    backend = backend or os.environ.get("EMBEDDING_BACKEND", "auto")
    return dims.get(backend, 768)
