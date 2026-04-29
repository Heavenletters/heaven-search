# Heavenletters Search

Semantic search engine for the [Heavenletters](https://heavenletters.org) corpus — 6,617 spiritual channeled messages spanning 20+ years.

## What it does

Three search modes over the full Heavenletters corpus:

- **Semantic** — meaning-based search using vector embeddings. Find Heavenletters by theme, feeling, or concept even without exact word matches.
- **Keyword** — traditional full-text search via SQLite FTS5.
- **Hybrid** — reciprocal rank fusion of both, weighted 70/30 toward semantic.

Plus an **`/analyze`** endpoint: ask a question, the engine retrieves relevant Heavenletters, and an LLM synthesises an answer grounded in the actual text.

## Architecture

```
FastAPI (port 8000)
├── src/main.py        — API routes: /search, /analyze, /stats, /auth/login
├── src/search.py      — Semantic, keyword, hybrid search with RRF
├── src/embedder.py    — sentence-transformers (all-MiniLM-L6-v2, 384-dim)
├── src/store.py       — SQLite + FTS5 + numpy vector matrix (~10MB)
├── src/ingest.py      — Batch JSON ingestion with embedding
└── src/auth.py        — JWT + static API key auth
```

Runs on CPU. No GPU needed. Designed for resource-constrained environments.

## Quick start

```bash
# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Build and run
docker compose up -d

# Check it's working
curl http://localhost:8000/health
```

### Ingesting Heavenletters

```bash
# From inside the container or with venv active
python -m src.ingest data/batch_001.json
python -m src.ingest data/batch_*.json
```

Expected format — JSON array:

```json
[
  {
    "id": "HL-0001",
    "title": "God Speaks",
    "content": "Full text of the Heavenletter..."
  }
]
```

### API usage

```bash
# Get auth token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"yourpassword"}' | jq -r .access_token)

# Search
curl "http://localhost:8000/search?q=divine+love+in+everyday+life&mode=hybrid&top=5" \
  -H "Authorization: Bearer $TOKEN"

# Analyze with LLM
curl -X POST http://localhost:8000/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What do Heavenletters say about finding purpose?", "top_k": 5}'
```

## Integration

Designed to sit behind an Astro or Keystone.js frontend. See [docs/INTEGRATION.md](docs/INTEGRATION.md) for wiring guides.

## Requirements

- Python 3.12+
- ~80MB for the embedding model (auto-downloaded at build)
- ~10MB for the vector index (6,617 documents × 384 dimensions)
- Docker recommended (2GB memory limit)

## License

Proprietary — All rights reserved. Heavenletters content is copyright Godwriting International Society of Heaven Ministries.
