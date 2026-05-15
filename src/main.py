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
import re
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


# ── Share / Permalink ─────────────────────────────────────────────

import hashlib
import secrets

SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "shared")
os.makedirs(SHARED_DIR, exist_ok=True)


class ShareRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    results: list[dict] = []
    analysis: str = ""
    sources: list[dict] = []
    backend: str = ""


class ShareResponse(BaseModel):
    url: str
    slug: str


@app.post("/share", response_model=ShareResponse)
def api_share(req: ShareRequest, _user: str = Depends(require_auth)):
    """Save search + analysis as a shareable permalink."""
    slug = secrets.token_hex(4)  # 8-char random hex

    # Build a self-contained HTML page
    results_html = _build_results_html(req.results, req.backend)
    analysis_html = _build_analysis_html(req.analysis, req.sources)

    page = SHARE_TEMPLATE.format(
        query=req.query,
        mode=req.mode,
        backend=req.backend,
        results_html=results_html,
        analysis_html=analysis_html,
        slug=slug,
    )

    path = os.path.join(SHARED_DIR, f"{slug}.html")
    with open(path, "w") as f:
        f.write(page)

    return ShareResponse(url=f"/share/{slug}", slug=slug)


@app.get("/share/{slug}")
def api_get_share(slug: str):
    """Serve a previously shared permalink."""
    # Sanitize slug to prevent path traversal
    safe = ''.join(c for c in slug if c.isalnum())
    if safe != slug:
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.join(SHARED_DIR, f"{safe}.html")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


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


# ── Share template & helpers ───────────────────────────────────────

def _html_escape(s):
    """Basic HTML entity escaping."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _build_results_html(results: list[dict], backend: str) -> str:
    if not results:
        return '<div class="status">No results</div>'
    parts = []
    for r in results:
        title = _html_escape(r.get("title", "Heavenletter"))
        excerpt = _html_escape(r.get("excerpt", ""))
        score = r.get("_score", r.get("score", 0))
        source = _html_escape(r.get("_source", ""))
        meta = r.get("metadata", {})
        pub_num = meta.get("publish_number", "")
        permalink = meta.get("permalink", "")
        url = f"https://heavenletters.org/{permalink}" if permalink else "#"

        parts.append(f'''<div class="result">
  <div class="result-header">
    <span class="result-title"><a href="{_html_escape(url)}" target="_blank">{title}</a></span>
    <span class="result-number">#{_html_escape(str(pub_num))}</span>
  </div>
  <div class="result-excerpt">{excerpt}</div>
  <div class="result-meta">
    <span>{source}</span>
    <span>score: {score:.4f}</span>
    <span>{_html_escape(backend)}</span>
  </div>
</div>''')
    return "\n".join(parts)


def _build_analysis_html(analysis: str, sources: list[dict]) -> str:
    if not analysis:
        return ""
    parts = ['<div id="analyze-result">', '<h3>Analysis</h3>']
    if sources:
        src_links = []
        for s in sources:
            num = _html_escape(str(s.get("publish_number", "")))
            title = _html_escape(s.get("title", ""))
            permalink = s.get("permalink", "")
            url = f"https://heavenletters.org/{permalink}" if permalink else "#"
            src_links.append(
                f'<span class="source-item"><a href="{_html_escape(url)}" target="_blank">'
                f'Heavenletter #{num}: {title}</a></span>'
            )
        parts.append('<div class="source-list">Based on: ' + ", ".join(src_links) + "</div>")
    # Wrap paragraphs for markdown-like text (basic rendering even without JS)
    for line in analysis.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("###"):
            parts.append(f"<h4>{_html_escape(line[3:].strip())}</h4>")
        elif line.startswith("##"):
            parts.append(f"<h3>{_html_escape(line[2:].strip())}</h3>")
        elif line.startswith("#"):
            parts.append(f"<h3>{_html_escape(line[1:].strip())}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            parts.append(f"<li>{_html_escape(line[2:])}</li>")
        elif line.startswith("> "):
            parts.append(f"<blockquote>{_html_escape(line[2:])}</blockquote>")
        else:
            # Bold markers
            line = _html_escape(line)
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            parts.append(f"<p>{line}</p>")
    parts.append("</div>")
    return "\n".join(parts)


SHARE_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Heavenletters: {query}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Lexend:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --cream: #f9f0d9; --white: #ffffff; --charcoal: #403f3e;
    --ab-standard: #2e82f5; --ab-dark: #084eaf; --ab-light: #d8e8fd;
    --mb-standard: #ED2960; --mb-dark: #bb1141; --mb-light: #f8b4c7;
    --bl-standard: #FDE80F; --bl-light: #fef9c2; --beige: #b7b2a3;
  }}
  body {{
    font-family: 'Lexend', sans-serif;
    background: var(--cream); color: var(--charcoal);
    font-size: 1.125rem; line-height: 1.75; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
  }}
  h1, h2, h3, h4 {{ font-family: 'EB Garamond', serif; }}
  header {{
    width: 100%; padding: 3rem 1.5rem 0; text-align: center;
  }}
  header h1 {{
    font-size: 2rem; font-weight: 600; letter-spacing: 0.02em; color: var(--mb-standard);
  }}
  header p {{ font-size: 0.95rem; color: var(--charcoal); opacity: 0.6; margin-top: 0.35rem; }}
  main {{
    width: 100%; max-width: 760px; padding: 2rem 1.5rem 4rem;
  }}
  .query-display {{
    background: var(--white); border: 1px solid var(--beige);
    border-radius: 0.6rem; padding: 1.5rem 1.75rem;
    margin-bottom: 2rem;
  }}
  .query-display .q {{ font-family: 'EB Garamond', serif; font-size: 1.35rem; color: var(--charcoal); }}
  .query-display .meta {{ font-size: 0.85rem; color: var(--beige); margin-top: 0.5rem; }}
  .result {{
    background: var(--white); border: 1px solid var(--beige);
    border-radius: 0.6rem; padding: 1.5rem 1.75rem; margin-bottom: 1rem;
  }}
  .result-header {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 0.6rem; gap: 1rem;
  }}
  .result-title {{ font-family: 'EB Garamond', serif; font-size: 1.2rem; font-weight: 600; }}
  .result-title a {{ color: var(--charcoal); text-decoration: none; }}
  .result-title a:hover {{ color: var(--mb-standard); }}
  .result-number {{ font-size: 0.9rem; color: var(--beige); white-space: nowrap; }}
  .result-excerpt {{ font-size: 1rem; line-height: 1.7; opacity: 0.85; }}
  .result-meta {{
    display: flex; gap: 1rem; margin-top: 0.75rem;
    font-size: 0.8rem; color: var(--beige);
  }}
  .result-meta span {{
    background: var(--cream); padding: 0.15rem 0.5rem; border-radius: 0.3rem;
  }}
  #analyze-result {{
    margin-top: 3rem; background: var(--white);
    border: 1px solid var(--beige); border-left: 4px solid var(--mb-standard);
    padding: 2rem; border-radius: 0 0.6rem 0.6rem 0;
    font-size: 1rem; line-height: 1.75;
  }}
  #analyze-result h3 {{
    font-family: 'EB Garamond', serif; font-size: 1.5rem; font-weight: 600;
    color: var(--mb-standard); margin-bottom: 1rem;
  }}
  #analyze-result h4 {{
    font-family: 'EB Garamond', serif; font-weight: 600;
    margin: 1.25rem 0 0.5rem; font-size: 1.15rem;
  }}
  #analyze-result p {{ margin-bottom: 0.65rem; }}
  #analyze-result strong {{ color: var(--ab-dark); font-weight: 600; }}
  #analyze-result blockquote {{
    border-left: 3px solid var(--bl-standard);
    padding-left: 1rem; opacity: 0.75; margin: 0.75rem 0;
    font-style: italic; font-family: 'EB Garamond', serif;
  }}
  #analyze-result li {{ margin: 0.15rem 0 0.15rem 1.5rem; }}
  .source-list {{
    margin-bottom: 1.25rem; padding-bottom: 1rem;
    border-bottom: 1px solid var(--beige); font-size: 0.95rem; opacity: 0.75;
  }}
  .source-item a {{ color: var(--ab-standard); text-decoration: none; }}
  .source-item a:hover {{ text-decoration: underline; }}
  .status {{ text-align: center; opacity: 0.5; padding: 3rem 0; font-size: 1.05rem; }}
  .footer {{
    text-align: center; padding: 1.5rem; font-size: 0.85rem;
    color: var(--beige); font-family: 'Lexend', sans-serif;
  }}
  .footer a {{ color: var(--ab-standard); text-decoration: none; }}
</style>
</head>
<body>
<header>
  <h1>Heavenletters Search</h1>
  <p>Shared result</p>
</header>
<main>
<div class="query-display">
  <div class="q">"{query}"</div>
  <div class="meta">{mode} &middot; {backend}</div>
</div>
{results_html}
{analysis_html}
</main>
<div class="footer">
  <a href="/">Search Heavenletters</a>
</div>
</body>
</html>'''


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
