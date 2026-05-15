"""CLI for querying the Heavenletters search engine.

Usage:
    hl search "creating community" --top 5
    hl search "divine will" --mode hybrid --top 10
    hl stats
    hl get 42              # retrieve document by ID
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Auto-load .env for CLI usage
from dotenv import load_dotenv
load_dotenv()

from .embedder import AutoEmbedder, get_embedder
from .search import hybrid_search, keyword_search, semantic_search
from .store import DEFAULT_EMBEDDING_DIM, DocStore


def _get_stores(data_dir: str):
    """Return (vertex_store, local_store, primary_store, embedder)."""
    vertex_store = DocStore(data_dir, dim=DEFAULT_EMBEDDING_DIM, vec_suffix="vertex")
    local_store = DocStore(data_dir, dim=DEFAULT_EMBEDDING_DIM, vec_suffix="local")

    # If Vertex vectors exist, use auto (Vertex primary, local fallback)
    if vertex_store.has_vectors and vertex_store.dim == 768:
        embedder = get_embedder("auto")
    # If only legacy MiniLM (384-dim) exists, use that directly
    elif local_store.has_vectors and local_store.dim == 384:
        embedder = get_embedder("minilm")
    # Fall back to configured backend
    else:
        embedder = get_embedder()

    primary_store = vertex_store if vertex_store.has_vectors else local_store
    return vertex_store, local_store, primary_store, embedder


def cmd_search(args):
    vertex_store, local_store, primary_store, embedder = _get_stores(args.data_dir)

    if vertex_store.count() == 0:
        print("No documents in the database. Ingest some JSON batches first:")
        print("  python -m src.ingest data/*.json")
        vertex_store.close()
        local_store.close()
        return

    if args.mode == "semantic":
        results = semantic_search(
            primary_store, args.query, top_k=args.top,
            embedder=embedder, fallback_store=local_store,
        )
    elif args.mode == "keyword":
        results = keyword_search(vertex_store, args.query, top_k=args.top)
    else:
        results = hybrid_search(
            primary_store, args.query, top_k=args.top,
            embedder=embedder, fallback_store=local_store,
        )

    # Detect backend used
    backend = "unknown"
    if isinstance(embedder, AutoEmbedder):
        backend = "vertex" if embedder.primary_healthy else "local"

    if not results:
        print(f"No results for: \"{args.query}\"")
        vertex_store.close()
        local_store.close()
        return

    print(f"\n{'─'*70}")
    print(f" Results for: \"{args.query}\"  ({args.mode}, top {args.top})  [{backend}]")
    print(f"{'─'*70}\n")

    for i, doc in enumerate(results, 1):
        title = doc.get("title") or f"#{doc['id']}"
        score = doc.get("_score", "?")
        source = doc.get("_source", "?")

        meta = doc.get("metadata", {})
        pub_num = meta.get("publish_number", "Unknown")
        permalink = meta.get("permalink", "")

        print(f"  [{i}] Title: {title} | Publish Number: {pub_num}")
        print(f"      score: {score}  |  source: {source}")
        if permalink:
            print(f"      url: https://heavenletters.org/{permalink}")

        excerpt = doc.get("_excerpt")
        if not excerpt:
            content = doc["content"]
            excerpt = content[:200].replace("\n", " ").strip()
            if len(content) > 200:
                excerpt += "…"

        print(f"      {excerpt}")
        print()

    vertex_store.close()
    local_store.close()


def cmd_get(args):
    vertex_store, local_store, _, _ = _get_stores(args.data_dir)
    doc = vertex_store.get_document(args.doc_id)
    if doc is None:
        print(f"Document #{args.doc_id} not found.")
    else:
        print(f"\n{'─'*70}")
        title = doc.get("title") or f"Document #{doc['id']}"
        print(f" {title}")
        if doc.get("external_id"):
            print(f" External ID: {doc['external_id']}")
        print(f"{'─'*70}\n")
        print(doc["content"])
        print()
        if doc.get("metadata"):
            print(f"Metadata: {doc['metadata']}")
    vertex_store.close()
    local_store.close()


def cmd_stats(args):
    vertex_store, local_store, _, _ = _get_stores(args.data_dir)
    n = vertex_store.count()

    def _vec_info(label, store):
        try:
            if store.has_vectors:
                path = store.vec_path
                if not path.exists():
                    path = store._legacy_path
                size_mb = path.stat().st_size / (1024 * 1024)
                return f"  {label}: {size_mb:.1f} MB ({store.dim}-dim)"
            return f"  {label}: not ingested"
        except (FileNotFoundError, OSError):
            return f"  {label}: not ingested"

    db_size_mb = (
        vertex_store.db_path.stat().st_size / (1024 * 1024)
        if vertex_store.db_path.exists() else 0
    )

    print(f"\n  Documents:    {n}")
    print(f"  SQLite DB:    {db_size_mb:.1f} MB")
    print(_vec_info("Vertex", vertex_store))
    print(_vec_info("Local ", local_store))
    print()
    vertex_store.close()
    local_store.close()


def main():
    parser = argparse.ArgumentParser(description="Heavenletters Semantic Search")
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search Heavenletters")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--top", type=int, default=10, help="Number of results")
    search_p.add_argument(
        "--mode",
        choices=["hybrid", "semantic", "keyword"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    search_p.add_argument("--data-dir", default="data", help="Data directory")

    get_p = sub.add_parser("get", help="Retrieve a document by ID")
    get_p.add_argument("doc_id", type=int, help="Document internal ID")
    get_p.add_argument("--data-dir", default="data", help="Data directory")

    stats_p = sub.add_parser("stats", help="Show database statistics")
    stats_p.add_argument("--data-dir", default="data", help="Data directory")

    args = parser.parse_args()
    if args.command == "search":
        cmd_search(args)
    elif args.command == "get":
        cmd_get(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
