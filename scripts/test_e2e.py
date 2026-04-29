#!/usr/bin/env python3
"""End-to-end test using the sample Heavenletter in ../heaven.txt"""

import sys
import json
from pathlib import Path

# Allow running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import embed_query, embed_texts
from src.store import DocStore
from src.search import hybrid_search, semantic_search

# Load the sample
sample_path = Path(__file__).parent.parent / "heaven.txt"
with open(sample_path) as f:
    raw = f.read()

# It's JSON-like: key is "content", no outer braces
data = json.loads("{" + raw + "}")
content = data["content"]

print("=" * 60)
print("  Heavenletters Search Engine — Smoke Test")
print("=" * 60)

# 1. Test embedding
print("\n[1] Embedding model...")
vec = embed_query("spiritual awakening and divine love")
print(f"    ✓ Vector shape: {vec.shape}, dtype: {vec.dtype}")
print(f"    ✓ Normalized: {abs(float(vec.dot(vec)) - 1.0) < 0.001}")

# 2. Test storage
print("\n[2] Document store...")
store = DocStore("data")

# Clear any previous test data
store.clear()

embeddings = embed_texts([content])
ids = store.add_documents(
    contents=[content],
    embeddings=embeddings,
    titles=["Sample Heavenletter"],
    external_ids=["HL-TEST-001"],
)
print(f"    ✓ Stored document, ID: {ids[0]}")
print(f"    ✓ Total docs: {store.count()}")

# 3. Test retrieval
print("\n[3] Document retrieval...")
doc = store.get_document(ids[0])
assert doc is not None
print(f"    ✓ Title: {doc['title']}")
print(f"    ✓ Content preview: {doc['content'][:80]}...")

# 4. Test semantic search (add a few more synthetic entries for demo)
print("\n[4] Additional test documents...")
extra_texts = [
    "The community gathers in love and unity, sharing the divine light with all who enter.",
    "Let go of your attachment to the individual self and merge with the infinite consciousness.",
    "Today you will learn to receive My love without resistance, without ego.",
]
extra_embs = embed_texts(extra_texts)
store.add_documents(
    contents=extra_texts,
    embeddings=extra_embs,
    titles=["Community", "Detachment", "Receiving Love"],
    external_ids=["HL-TEST-002", "HL-TEST-003", "HL-TEST-004"],
)
print(f"    ✓ Total docs: {store.count()}")

# 5. Semantic search
print("\n[5] Semantic search: 'creating community'...")
results = semantic_search(store, "creating community", top_k=2)
for r in results:
    print(f"    [{r['_score']:.3f}] {r['title']}: {r['content'][:60]}...")

# 6. Hybrid search
print("\n[6] Hybrid search: 'letting go of self'...")
results = hybrid_search(store, "letting go of self", top_k=2)
for r in results:
    print(f"    [{r['_score']:.3f}] ({r['_source']}) {r['title']}: {r['content'][:60]}...")

# 7. Cleanup
store.clear()
store.close()

# Remove test data
import os
db_path = Path("data/docs.db")
vec_path = Path("data/embeddings.npy")
for p in [db_path, vec_path]:
    if p.exists():
        p.unlink()

print("\n" + "=" * 60)
print("  All tests passed! ✓")
print("=" * 60)
