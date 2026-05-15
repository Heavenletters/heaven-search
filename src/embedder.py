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
    """Google text-embedding-004 via Vertex AI predict endpoint.

    Uses API key auth with x-goog-api-key header.
    Requires VERTEX_PROJECT and VERTEX_API_KEY env vars.
    Batch endpoint supports up to 250 instances per request.
    """

    dim = 768
    MODEL = "text-embedding-004"

    def __init__(
        self,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("VERTEX_API_KEY") or os.environ.get("VERTEX_AI_API_KEY", "")
        self.project = project or os.environ.get("VERTEX_PROJECT", "")
        self.location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self.max_batch = 250  # Vertex AI limit per predict call

    @property
    def available(self) -> bool:
        return bool(self.api_key) and bool(self.project)

    @property
    def _endpoint(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com"
            f"/v1/projects/{self.project}"
            f"/locations/{self.location}"
            f"/publishers/google/models/{self.MODEL}:predict"
        )

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import urllib.request
        import json

        if not self.api_key:
            raise RuntimeError("VERTEX_API_KEY not set")
        if not self.project:
            raise RuntimeError("VERTEX_PROJECT not set")

        all_embeddings: list[np.ndarray] = []

        # Adaptive batching: text-embedding-004 has a 20,000 token limit per request.
        # Estimate ~3.5 chars per token, keep batches under ~18K estimated tokens.
        MAX_TOKENS_PER_BATCH = 18000
        CHARS_PER_TOKEN = 3.5

        i = 0
        while i < len(texts):
            batch: list[str] = []
            est_tokens = 0
            while i < len(texts) and est_tokens < MAX_TOKENS_PER_BATCH:
                batch.append(texts[i])
                est_tokens += len(texts[i]) / CHARS_PER_TOKEN
                i += 1
            embeddings = self._embed_batch(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import urllib.request
        import json

        instances = [{"content": t} for t in texts]
        body = json.dumps({"instances": instances}).encode("utf-8")

        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Vertex API request failed: {e}")

        if "error" in data:
            raise RuntimeError(f"Vertex API error: {data['error']}")

        predictions = data.get("predictions", [])
        if len(predictions) != len(texts):
            raise RuntimeError(
                f"Vertex returned {len(predictions)} predictions for {len(texts)} texts"
            )

        embeddings = []
        for pred in predictions:
            values = pred.get("embeddings", {}).get("values", [])
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
