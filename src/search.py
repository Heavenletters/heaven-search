"""Semantic + keyword hybrid search over the document store."""

from __future__ import annotations

import numpy as np

from .embedder import embed_query
from .store import DocStore


def semantic_search(
    store: DocStore,
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[dict]:
    """Embed query and return top-k documents by cosine similarity.

    Uses brute-force dot product against the (N, 384) vector matrix.
    On 6,617 documents this takes <500μs on any hardware made after 2005.
    """
    if len(store.vectors) == 0:
        return []

    q_vec = embed_query(query)  # (384,), already normalized
    scores = np.dot(store.vectors, q_vec)  # (N,) cosine similarities

    # Get top-k indices
    if top_k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        # Partial sort — faster for small k
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            continue
        doc = store.get_document(idx + 1)  # internal IDs are 1-based
        if doc:
            doc["_score"] = round(score, 4)
            doc["_source"] = "semantic"
            results.append(doc)

    return results


def keyword_search(
    store: DocStore,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """FTS5 keyword search."""
    results = store.keyword_search(query, limit=top_k)
    for r in results:
        r["_source"] = "keyword"
        # Normalize rank to a pseudo-score (higher = better)
        r["_score"] = round(1.0 / (abs(r.get("_rank", 1)) + 1), 4)
    return results


def hybrid_search(
    store: DocStore,
    query: str,
    top_k: int = 10,
    semantic_weight: float = 0.7,
) -> list[dict]:
    """Combine semantic and keyword results with reciprocal rank fusion.

    Retrieves top_k * 3 from each source, fuses with RRF, returns top_k.
    """
    fetch_k = max(top_k * 3, 20)

    semantic_results = semantic_search(store, query, top_k=fetch_k)
    keyword_results = keyword_search(store, query, top_k=fetch_k)

    # Reciprocal Rank Fusion
    scores: dict[int, float] = {}
    docs: dict[int, dict] = {}

    k = 60  # RRF constant

    for rank, doc in enumerate(semantic_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + semantic_weight / (k + rank + 1)
        docs[doc_id] = doc

    for rank, doc in enumerate(keyword_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + (1 - semantic_weight) / (k + rank + 1)
        if doc_id not in docs:
            docs[doc_id] = doc

    # Sort by fused score
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

    results = []
    for doc_id in sorted_ids:
        doc = docs[doc_id]
        doc["_score"] = round(scores[doc_id], 4)
        doc["_source"] = "hybrid"
        results.append(doc)

    return results
