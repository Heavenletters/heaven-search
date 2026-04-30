"""FastAPI server exposing the Heavenletters search engine as a REST API.

Endpoints:
    POST /auth/login         — get JWT token
    GET  /search             — semantic/hybrid search
    GET  /documents/{id}    — retrieve document
    GET  /stats              — database stats
    POST /analyze            — search + LLM analysis (requires DEEPSEEK_API_KEY)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import LoginRequest, TokenResponse, login, require_auth
from .search import hybrid_search, keyword_search, semantic_search
from .store import DocStore

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

_store: DocStore | None = None


def get_store() -> DocStore:
    global _store
    if _store is None:
        _store = DocStore("data")
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    get_store()
    yield
    # Shutdown
    if _store:
        _store.close()


app = FastAPI(
    title="Heavenletters Search API",
    description="Semantic search over the Heavenletters corpus",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow web UI on any origin in dev, lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    id: int
    external_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    metadata: dict = {}
    score: float
    source: str  # "semantic", "keyword", or "hybrid"
    # Heavenletter-specific fields (extracted from metadata for convenience)
    permalink: Optional[str] = None
    publish_number: Optional[int] = None
    published_date: Optional[str] = None
    excerpt: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_docs: int
    results: list[SearchResult]


class AnalyzeRequest(BaseModel):
    query: str
    instructions: str = "Analyze the following documents and answer the user's question."
    top_k: int = 5
    model: str = "deepseek-chat"  # or deepseek-reasoner for deep thinking


class AnalyzeResponse(BaseModel):
    query: str
    model: str
    documents_retrieved: int
    tokens_used: Optional[int] = None
    analysis: str


# ── Routes ──────────────────────────────────────────────────────────

@app.post("/auth/login", response_model=TokenResponse)
def auth_login(req: LoginRequest):
    """Get a JWT token for API access."""
    return login(req)


@app.get("/search", response_model=SearchResponse)
def api_search(
    q: str = Query(..., description="Search query"),
    mode: str = Query("hybrid", pattern="^(hybrid|semantic|keyword)$"),
    top: int = Query(10, ge=1, le=100),
    _user: str = Depends(require_auth),
):
    """Search the Heavenletters corpus."""
    store = get_store()
    total = store.count()

    if mode == "semantic":
        results = semantic_search(store, q, top_k=top)
    elif mode == "keyword":
        results = keyword_search(store, q, top_k=top)
    else:
        results = hybrid_search(store, q, top_k=top)

    return SearchResponse(
        query=q,
        mode=mode,
        total_docs=total,
        results=[SearchResponse._scrub(r) for r in results],
    )


@app.get("/documents/{doc_id}")
def api_get_document(
    doc_id: int,
    _user: str = Depends(require_auth),
):
    """Retrieve a single document by internal ID."""
    store = get_store()
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/stats")
def api_stats(_user: str = Depends(require_auth)):
    """Get database statistics."""
    store = get_store()
    import os
    from pathlib import Path

    data_dir = Path("data")
    vec_path = data_dir / "embeddings.npy"
    return {
        "total_documents": store.count(),
        "db_size_mb": round(store.db_path.stat().st_size / (1024 * 1024), 2) if store.db_path.exists() else 0,
        "vector_size_mb": round(vec_path.stat().st_size / (1024 * 1024), 2) if vec_path.exists() else 0,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def api_analyze(
    req: AnalyzeRequest,
    _user: str = Depends(require_auth),
):
    """Search + LLM analysis. Retrieves top-k documents and passes them to DeepSeek."""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")

    store = get_store()
    results = hybrid_search(store, req.query, top_k=req.top_k)

    if not results:
        return AnalyzeResponse(
            query=req.query,
            model=req.model,
            documents_retrieved=0,
            analysis="No relevant documents found.",
        )

    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(results, 1):
        title = doc.get("title") or f"Heavenletter #{doc.get('external_id', doc['id'])}"
        context_parts.append(f"--- Document {i}: {title} ---\n{doc['content']}\n")

    context = "\n".join(context_parts)

    system_prompt = (
        "You are a research assistant analyzing a collection of Heavenletters — "
        "spiritual channeled messages that explore themes of divine love, unity, "
        "consciousness, and the human experience. "
        "You answer questions based on the documents provided. "
        "Be thorough, nuanced, and quote relevant passages. "
        "If the documents don't address the question, say so honestly."
    )

    user_message = f"{req.instructions}\n\nUser question: {req.query}\n\nRelevant documents:\n\n{context}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": req.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.3,
                "max_tokens": 4000,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]
    usage = data.get("usage", {})

    return AnalyzeResponse(
        query=req.query,
        model=req.model,
        documents_retrieved=len(results),
        tokens_used=usage.get("total_tokens"),
        analysis=choice["message"]["content"],
    )


# ── Health check (no auth required) ─────────────────────────────────

@app.get("/health")
def health():
    store = get_store()
    return {"status": "ok", "documents": store.count()}


# ── Helpers ─────────────────────────────────────────────────────────

# Attach helper to clean up result dicts for Pydantic serialization
@staticmethod
def _scrub(result: dict) -> SearchResult:
    meta = result.get("metadata", {})
    # publish_number might be stored as int or string in metadata
    pn = meta.get("publish_number")
    if pn is not None:
        try:
            pn = int(pn)
        except (ValueError, TypeError):
            pn = None
    return SearchResult(
        id=result["id"],
        external_id=result.get("external_id"),
        title=result.get("title"),
        content=result["content"],
        metadata=meta,
        score=result.get("_score", 0.0),
        source=result.get("_source", "unknown"),
        permalink=meta.get("permalink"),
        publish_number=pn,
        published_date=meta.get("published_date"),
        excerpt=result.get("_excerpt"),
    )


SearchResponse._scrub = _scrub  # type: ignore
