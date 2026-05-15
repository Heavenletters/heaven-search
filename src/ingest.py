"""Ingest Heavenletters from JSON batches.

Expected format — one JSON file per batch of 100:

    [
        {
            "id": "HL-0001",          // optional — external ID
            "title": "The Topic",     // optional
            "content": "Full text...", // required
            ...any other fields...     // stored in metadata
        },
        ...
    ]

Usage:
    python -m src.ingest data/batch_001.json --backend vertex
    python -m src.ingest data/batch_*.json --backend local
    python -m src.ingest data/batch_*.json --backend vertex  # re-ingest for new model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .embedder import get_embedder, get_dim
from .store import DocStore


def load_batch(path: Path) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def ingest_batch(
    store: DocStore,
    documents: list[dict],
    batch_size: int = 32,
    backend: str = "vertex",
):
    """Embed and store a batch of documents using the specified backend."""
    embedder = get_embedder(backend)
    contents = []
    external_ids = []
    titles = []
    metadata_list = []

    for doc in documents:
        content = doc.get("content", doc.get("text", doc.get("body", "")))
        if not content:
            print(f"  ⚠ skipping document with empty content: {doc.get('id', doc.get('nid', '?'))}")
            continue
        contents.append(content)

        ext_id = doc.get("id", doc.get("nid", doc.get("publish_number", None)))
        title = doc.get("title", None)
        meta = {k: v for k, v in doc.items() if k not in ("id", "title", "content", "text")}

        external_ids.append(str(ext_id) if ext_id is not None else None)
        titles.append(title)
        metadata_list.append(meta)

    if not contents:
        print("  No valid documents in batch.")
        return

    dim = embedder.dim
    print(f"  Embedding {len(contents)} documents with {backend} ({dim}-dim)...")
    embeddings = embedder.embed_texts(contents, batch_size=batch_size)

    print(f"  Storing to database ({store.vec_path.name})...")
    ids = store.add_documents(
        contents=contents,
        embeddings=embeddings,
        external_ids=external_ids,
        titles=titles,
        metadata_list=metadata_list,
    )
    print(f"  ✓ Stored {len(ids)} documents (IDs: {ids[0]} — {ids[-1]})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <json_file> [json_file ...] [--backend vertex|local|minilm]")
        print("       python -m src.ingest data/batch_*.json --backend vertex")
        sys.exit(1)

    # Parse backend flag
    args = sys.argv[1:]
    backend = "vertex"
    paths_args = []
    for arg in args:
        if arg.startswith("--backend="):
            backend = arg.split("=", 1)[1]
        elif arg == "--backend":
            backend = "vertex"  # handled below
        else:
            paths_args.append(arg)

    # Expand glob patterns
    paths: list[Path] = []
    for arg in paths_args:
        expanded = sorted(Path().glob(arg))
        if expanded:
            paths.extend(expanded)
        else:
            p = Path(arg)
            if p.exists():
                paths.append(p)
            else:
                print(f"  ⚠ file not found: {arg}")

    if not paths:
        print("No valid files found.")
        sys.exit(1)

    dim = get_dim(backend)
    vec_suffix = "vertex" if backend == "vertex" else "local"
    store = DocStore("data", dim=dim, vec_suffix=vec_suffix)

    print(f"Backend: {backend} ({dim}-dim)")
    print(f"Vector file: {store.vec_path}")
    print(f"Database: {store.db_path}")
    print(f"Documents before: {store.count()}\n")

    for path in paths:
        print(f"Processing: {path.name}")
        try:
            docs = load_batch(path)
            ingest_batch(store, docs, backend=backend)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDocuments after: {store.count()}")
    store.close()


if __name__ == "__main__":
    main()
