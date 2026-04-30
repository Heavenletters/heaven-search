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

from .search import hybrid_search, keyword_search, semantic_search
from .store import DocStore


def cmd_search(args):
    store = DocStore(args.data_dir)

    if store.count() == 0:
        print("No documents in the database. Ingest some JSON batches first:")
        print("  python -m src.ingest data/*.json")
        store.close()
        return

    if args.mode == "semantic":
        results = semantic_search(store, args.query, top_k=args.top)
    elif args.mode == "keyword":
        results = keyword_search(store, args.query, top_k=args.top)
    else:
        results = hybrid_search(store, args.query, top_k=args.top)

    if not results:
        print(f"No results for: \"{args.query}\"")
        store.close()
        return

    print(f"\n{'─'*70}")
    print(f" Results for: \"{args.query}\"  ({args.mode}, top {args.top})")
    print(f"{'─'*70}\n")

    for i, doc in enumerate(results, 1):
        title = doc.get("title") or f"#{doc['id']}"
        score = doc.get("_score", "?")
        source = doc.get("_source", "?")
        
        # Grab metadata fields
        meta = doc.get("metadata", {})
        pub_num = meta.get("publish_number", "Unknown")
        permalink = meta.get("permalink", "")

        print(f"  [{i}] Title: {title} | Publish Number: {pub_num}")
        print(f"      score: {score}  |  source: {source}")
        if permalink:
            print(f"      url: https://heavenletters.org/{permalink}")

        # Show the smart excerpt if it exists, otherwise fall back to content preview
        excerpt = doc.get("_excerpt")
        if not excerpt:
            content = doc["content"]
            excerpt = content[:200].replace("\n", " ").strip()
            if len(content) > 200:
                excerpt += "…"
                
        print(f"      {excerpt}")
        print()

    store.close()


def cmd_get(args):
    store = DocStore(args.data_dir)
    doc = store.get_document(args.doc_id)
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
    store.close()


def cmd_stats(args):
    store = DocStore(args.data_dir)
    n = store.count()
    vec_path = Path(args.data_dir) / "embeddings.npy"
    vec_size_mb = vec_path.stat().st_size / (1024 * 1024) if vec_path.exists() else 0
    db_size_mb = store.db_path.stat().st_size / (1024 * 1024) if store.db_path.exists() else 0

    print(f"\n  Documents:    {n}")
    print(f"  Vector file:  {vec_size_mb:.1f} MB")
    print(f"  SQLite DB:    {db_size_mb:.1f} MB")
    print(f"  Total on disk: {(vec_size_mb + db_size_mb):.1f} MB")
    print()
    store.close()


def main():
    parser = argparse.ArgumentParser(description="Heavenletters Semantic Search")
    sub = parser.add_subparsers(dest="command")

    # search
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

    # get
    get_p = sub.add_parser("get", help="Retrieve a document by ID")
    get_p.add_argument("doc_id", type=int, help="Document internal ID")
    get_p.add_argument("--data-dir", default="data", help="Data directory")

    # stats
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
