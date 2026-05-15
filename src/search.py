"""Semantic + keyword hybrid search over the document store."""

from __future__ import annotations

import re

import numpy as np

from .embedder import embed_query
from .store import DocStore


# ── Excerpt generation ──────────────────────────────────────────────

def generate_excerpt(content: str, query: str, max_length: int = 300) -> str:
    """Extract the most relevant passage from content given a query.

    Strategy:
      1. Split content into sentences.
      2. Score each sentence by overlap with query terms.
      3. Return the best sentence ± surrounding context, up to max_length.
      4. Fall back to the first max_length chars if no good match.
    """
    # Normalise and tokenise query for matching
    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return _truncate(content, max_length)

    # Split into sentences (handles \r\n and \n)
    sentences = re.split(r"(?<=[.!?])\s+|\r?\n", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return _truncate(content, max_length)

    # Score each sentence
    best_idx = 0
    best_score = -1
    for i, sent in enumerate(sentences):
        sent_terms = set(re.findall(r"\w+", sent.lower()))
        overlap = len(query_terms & sent_terms)
        if overlap > best_score:
            best_score = overlap
            best_idx = i

    # If no overlap at all, use the beginning
    if best_score == 0:
        return _truncate(content, max_length)

    # Build excerpt from best sentence ± neighbours
    excerpt_parts: list[str] = []
    total_len = 0
    start = max(0, best_idx - 1)

    for i in range(start, len(sentences)):
        if total_len + len(sentences[i]) > max_length:
            break
        excerpt_parts.append(sentences[i])
        total_len += len(sentences[i]) + 1
        # Stop after including the best sentence + 1 after
        if i > best_idx:
            break

    excerpt = " ".join(excerpt_parts)
    if len(content) > len(excerpt) + 10:
        excerpt += "…"
    return excerpt


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "…"


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
            doc["_excerpt"] = generate_excerpt(doc["content"], query)
            results.append(doc)

    return results


# ── Query fragment extraction ──────────────────────────────────────

def _extract_fragments(query: str, max_fragments: int = 4) -> list[str]:
    """Break a long query into shorter sub-phrases for fallback search.

    When a user pastes a full sentence (especially a misquoted citation),
    the exact phrase won't match. This extracts progressively shorter
    trailing fragments — the assumption being that the most distinctive
    part of a quote is usually toward the end.

    Example: "The illusion created is that One ever existed as Two."
      → ["One ever existed as Two", "ever existed as Two", "existed as Two", "as Two"]
    """
    # Strip trailing punctuation
    clean = re.sub(r'[.!?,;:]+$', '', query.strip())
    words = clean.split()
    n = len(words)

    fragments = []
    # Try fragments of decreasing length, starting from ~6 words
    for frag_len in range(min(n, 6), 1, -1):
        # Take the last frag_len words (most distinctive tail)
        frag = ' '.join(words[-frag_len:])
        # Skip fragments that are identical to earlier (longer) ones or too short
        if len(frag.split()) >= 3 and frag not in fragments:
            fragments.append(frag)
        if len(fragments) >= max_fragments:
            break

    return fragments


def keyword_search(
    store: DocStore,
    query: str,
    top_k: int = 10,
    force_phrase: bool = False,
) -> list[dict]:
    """FTS5 keyword search with fallback cascade for long queries.

    If the full phrase query returns too few results, tries progressively
    shorter sub-phrases to catch near-matches (e.g. misquoted citations).
    Fragment results are scored lower than exact phrase matches.

    Set force_phrase=True to require adjacency (phrase matching) regardless
    of special characters in the query. Used for fragment fallback searches.
    """
    results = store.keyword_search(query, limit=top_k, force_phrase=force_phrase)
    seen_ids = set()
    for r in results:
        seen_ids.add(r["id"])
        r["_source"] = "keyword"
        r["_score"] = round(1.0 / (abs(r.get("_rank", 1)) + 1), 4)
        r["_match_type"] = "exact"
        r["_excerpt"] = generate_excerpt(r["content"], query)

    # Fallback cascade: if too few exact results, try shorter fragments
    min_results = 3
    if len(results) < min_results and len(query.split()) > 3:
        fragments = _extract_fragments(query)
        for i, frag in enumerate(fragments):
            # Use phrase matching for fragments — adjacency requirement makes
            # them far more selective. The cascade of different fragment lengths
            # already provides the fuzziness needed for slight misquotes.
            frag_results = store.keyword_search(frag, limit=top_k, force_phrase=True)
            # Score penalty: fragments get lower priority than exact matches.
            # Longer (more distinctive) fragments get less penalty.
            penalty = 0.6 - (i * 0.1)
            frag_label = frag
            for r in frag_results:
                doc_id = r["id"]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    r["_source"] = "keyword"
                    r["_score"] = round(
                        (1.0 / (abs(r.get("_rank", 1)) + 1)) * penalty, 4
                    )
                    r["_match_type"] = "fragment"
                    r["_excerpt"] = generate_excerpt(r["content"], query)
                    results.append(r)

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

    # Detect if keyword only found fragment matches (no exact phrase hits)
    has_exact_keyword = any(
        r.get("_match_type") == "exact" for r in keyword_results
    )
    # If keyword leg is all fragments, boost its weight — text matches
    # are more valuable than broad semantic similarity for citation queries
    kw_weight = (
        1 - semantic_weight
        if has_exact_keyword
        else max(1 - semantic_weight, 0.6)
    )
    sem_weight = 1.0 - kw_weight

    for rank, doc in enumerate(semantic_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + sem_weight / (k + rank + 1)
        docs[doc_id] = doc

    for rank, doc in enumerate(keyword_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + kw_weight / (k + rank + 1)
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
