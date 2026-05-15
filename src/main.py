"""FastAPI server exposing the Heavenletters search engine as a REST API.

Endpoints:
    POST /auth/login         — get JWT token
    GET  /search             — semantic/hybrid search (auto-failover)
    GET  /documents/{id}     — retrieve document
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .auth import LoginRequest, TokenResponse, login, require_auth
from .embedder import AutoEmbedder, get_embedder
from .search import hybrid_search, keyword_search, semantic_search
from .store import DEFAULT_EMBEDDING_DIM, DocStore

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
ANALYZE_MODEL = os.environ.get("ANALYZE_MODEL", "deepseek-v4-flash")
ANALYZE_SYSTEM_PROMPT = os.environ.get(
    "ANALYZE_SYSTEM_PROMPT",
    (
        "You are a research assistant analyzing a collection of Heavenletters — "
        "spiritual channeled messages that explore themes of divine love, unity, "
        "consciousness, and the human experience. "
        "You answer questions based on the documents provided. "
        "Be thorough, nuanced, and quote relevant passages. "
        "If the documents don't address the question, say so honestly."
    ),
)

# Dual stores: share the same SQLite DB, different .npy vector files
_vertex_store: DocStore | None = None
_local_store: DocStore | None = None
_embedder = None


def get_vertex_store() -> DocStore:
    global _vertex_store
    if _vertex_store is None:
        _vertex_store = DocStore("data", dim=DEFAULT_EMBEDDING_DIM, vec_suffix="vertex")
    return _vertex_store


def get_local_store() -> DocStore:
    global _local_store
    if _local_store is None:
        _local_store = DocStore("data", dim=DEFAULT_EMBEDDING_DIM, vec_suffix="local")
    return _local_store


def get_embedder_instance():
    global _embedder
    if _embedder is None:
        vertex_store = get_vertex_store()
        local_store = get_local_store()
        # If Vertex vectors exist, use auto (Vertex primary, local fallback)
        if vertex_store.has_vectors and vertex_store.dim == 768:
            _embedder = get_embedder("auto")
        # If only legacy MiniLM (384-dim) exists, use that directly
        elif local_store.has_vectors and local_store.dim == 384:
            _embedder = get_embedder("minilm")
        # Fall back to configured backend
        else:
            _embedder = get_embedder()
    return _embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_vertex_store()
    get_local_store()
    get_embedder_instance()
    yield
    if _vertex_store:
        _vertex_store.close()
    if _local_store:
        _local_store.close()


app = FastAPI(
    title="Heavenletters Search API",
    description="Semantic search over the Heavenletters corpus",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static UI ──────────────────────────────────────────────────────

UI_DIR = os.path.join(os.path.dirname(__file__), "ui")

if os.path.isdir(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    @app.get("/")
    async def serve_ui():
        return FileResponse(os.path.join(UI_DIR, "index.html"))


# ── Models ──────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    id: int
    external_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    metadata: dict = {}
    score: float
    source: str  # "semantic", "keyword", or "hybrid"
    permalink: Optional[str] = None
    publish_number: Optional[int] = None
    published_date: Optional[str] = None
    excerpt: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_docs: int
    results: list[SearchResult]
    embedding_backend: str = "unknown"


class AnalyzeRequest(BaseModel):
    query: str
    instructions: str = "Analyze the following documents and answer the user's question."
    top_k: int = 5
    model: str = ANALYZE_MODEL


class AnalyzeResponse(BaseModel):
    query: str
    model: str
    documents_retrieved: int
    tokens_used: Optional[int] = None
    analysis: str
    sources: list[dict] = []


# ── Routes ──────────────────────────────────────────────────────────

@app.post("/auth/login", response_model=TokenResponse)
def auth_login(req: LoginRequest):
    return login(req)


@app.get("/search", response_model=SearchResponse)
def api_search(
    q: str = Query(..., description="Search query"),
    mode: str = Query("hybrid", pattern="^(hybrid|semantic|keyword)$"),
    top: int = Query(10, ge=1, le=100),
    _user: str = Depends(require_auth),
):
    vertex_store = get_vertex_store()
    local_store = get_local_store()
    embedder = get_embedder_instance()
    total = vertex_store.count()

    # Determine which store to use for semantic search based on what's ingested
    primary_store = vertex_store if vertex_store.has_vectors else local_store

    if mode == "semantic":
        results = semantic_search(
            primary_store, q, top_k=top,
            embedder=embedder, fallback_store=local_store,
        )
    elif mode == "keyword":
        results = keyword_search(vertex_store, q, top_k=top)
    else:
        results = hybrid_search(
            primary_store, q, top_k=top,
            embedder=embedder, fallback_store=local_store,
        )

    # Detect which backend was actually used
    backend_used = "unknown"
    if isinstance(embedder, AutoEmbedder):
        backend_used = "vertex" if embedder.primary_healthy else "local"
    else:
        backend_used = type(embedder).__name__.lower().replace("embedder", "")

    return SearchResponse(
        query=q,
        mode=mode,
        total_docs=total,
        results=[SearchResponse._scrub(r) for r in results],
        embedding_backend=backend_used,
    )


@app.get("/documents/{doc_id}")
def api_get_document(
    doc_id: int,
    _user: str = Depends(require_auth),
):
    store = get_vertex_store()
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/stats")
def api_stats(_user: str = Depends(require_auth)):
    vertex_store = get_vertex_store()
    local_store = get_local_store()

    def _vec_info(store: DocStore) -> dict:
        try:
            if store.has_vectors:
                path = store.vec_path if store.vec_path.exists() else store._legacy_path
                size_mb = round(path.stat().st_size / (1024 * 1024), 2)
                return {"has_vectors": True, "dim": store.dim, "size_mb": size_mb}
        except (FileNotFoundError, OSError):
            pass
        return {"has_vectors": False, "dim": store.dim, "size_mb": 0}

    return {
        "total_documents": vertex_store.count(),
        "db_size_mb": (
            round(vertex_store.db_path.stat().st_size / (1024 * 1024), 2)
            if vertex_store.db_path.exists() else 0
        ),
        "vertex": _vec_info(vertex_store),
        "local": _vec_info(local_store),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def api_analyze(
    req: AnalyzeRequest,
    _user: str = Depends(require_auth),
):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")

    vertex_store = get_vertex_store()
    local_store = get_local_store()
    embedder = get_embedder_instance()
    primary_store = vertex_store if vertex_store.has_vectors else local_store

    results = hybrid_search(
        primary_store, req.query, top_k=req.top_k,
        embedder=embedder, fallback_store=local_store,
    )

    if not results:
        return AnalyzeResponse(
            query=req.query,
            model=req.model,
            documents_retrieved=0,
            analysis="No relevant documents found.",
        )

    context_parts = []
    sources = []
    for i, doc in enumerate(results, 1):
        pub_num = doc.get("metadata", {}).get("publish_number", "Unknown")
        title = doc.get("title") or f"Heavenletter"
        permalink = doc.get("metadata", {}).get("permalink", "")
        sources.append({
            "publish_number": pub_num,
            "title": title,
            "permalink": permalink,
        })
        label = f"Heavenletter #{pub_num}: {title}"
        context_parts.append(f"--- {label} ---\n{doc['content']}\n")

    context = "\n".join(context_parts)

    system_prompt = ANALYZE_SYSTEM_PROMPT

    user_message = f"{req.instructions}\n\nUser question: {req.query}\n\nRelevant documents:\n\n{context}"

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
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
        sources=sources,
    )


@app.get("/health")
def health():
    vertex_store = get_vertex_store()
    local_store = get_local_store()
    embedder = get_embedder_instance()
    backend = type(embedder).__name__.lower().replace("embedder", "")
    return {
        "status": "ok",
        "documents": vertex_store.count(),
        "vertex_ready": vertex_store.has_vectors,
        "local_ready": local_store.has_vectors,
        "backend": backend,
    }


# ── Helpers ─────────────────────────────────────────────────────────

@staticmethod
def _scrub(result: dict) -> SearchResult:
    meta = result.get("metadata", {})
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
