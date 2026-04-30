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
    python -m src.ingest data/batch_001.json
    python -m src.ingest data/batch_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .embedder import embed_texts
from .store import DocStore


def load_batch(path: Path) -> list[dict]:
    """Load a JSON batch file."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def ingest_batch(store: DocStore, documents: list[dict], batch_size: int = 32):
    """Embed and store a batch of documents."""
    contents = []
    external_ids = []
    titles = []
    metadata_list = []

    for doc in documents:
        # content is the only required field — supports "content", "text", or "body"
        content = doc.get("content", doc.get("text", doc.get("body", "")))
        if not content:
            print(f"  ⚠ skipping document with empty content: {doc.get('id', doc.get('nid', '?'))}")
            continue
        contents.append(content)

        # Extract known fields, dump rest into metadata
        # external ID: prefer "id", fall back to "nid" or "publish_number"
        ext_id = doc.get("id", doc.get("nid", doc.get("publish_number", None)))
        title = doc.get("title", None)
        meta = {k: v for k, v in doc.items() if k not in ("id", "title", "content", "text")}

        external_ids.append(str(ext_id) if ext_id is not None else None)
        titles.append(title)
        metadata_list.append(meta)

    if not contents:
        print("  No valid documents in batch.")
        return

    print(f"  Embedding {len(contents)} documents...")
    embeddings = embed_texts(contents, batch_size=batch_size)

    print(f"  Storing to database...")
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
        print("Usage: python -m src.ingest <json_file> [json_file ...]")
        print("       python -m src.ingest data/batch_*.json")
        sys.exit(1)

    # Expand glob patterns
    paths: list[Path] = []
    for arg in sys.argv[1:]:
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

    store = DocStore("data")
    print(f"Database: {store.db_path}")
    print(f"Documents before: {store.count()}\n")

    for path in paths:
        print(f"Processing: {path.name}")
        try:
            docs = load_batch(path)
            ingest_batch(store, docs)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print(f"\nDocuments after: {store.count()}")
    store.close()


if __name__ == "__main__":
    main()
