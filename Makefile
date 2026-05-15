.PHONY: install test dev build up down shell ingest ingest-vertex ingest-local password query

# Load .env if it exists
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Install dependencies in a virtual environment
install:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "✓ Installation complete."
	@echo "  Embedding models download on first use."

# Run the end-to-end test
test:
	./venv/bin/python scripts/test_e2e.py

# Start API server in development mode
dev:
	HEAVEN_API_KEY=dev-key ./venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Docker
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose exec heaven-search bash

# ── Ingestion ───────────────────────────────────────────────────────

# Ingest with Vertex AI (cloud) — best quality, needs VERTEX_API_KEY
# Usage: make ingest-vertex FILES="data/batch_*.json"
ingest-vertex:
	./venv/bin/python -m src.ingest $(FILES) --backend vertex

# Ingest with local bge-base-en-v1.5 (~438MB download, 768-dim)
# Usage: make ingest-local FILES="data/batch_*.json"
ingest-local:
	./venv/bin/python -m src.ingest $(FILES) --backend local

# Ingest with MiniLM (384-dim, legacy, ~80MB)
# Usage: make ingest-minilm FILES="data/batch_*.json"
ingest-minilm:
	./venv/bin/python -m src.ingest $(FILES) --backend minilm

# Backward-compatible: uses default backend (vertex)
# Usage: make ingest FILES="data/batch_001.json data/batch_002.json"
ingest:
	./venv/bin/python -m src.ingest $(FILES)

# ── Utilities ───────────────────────────────────────────────────────

# Generate a password hash for the admin user
password:
	@read -p "Admin password: " PASS; \
	./venv/bin/python scripts/hash_password.py "$$PASS"

# CLI search shortcut
# Usage: make query Q="creating community" TOP=5
query:
	./venv/bin/python -m src.cli search "$(Q)" --top $(or $(TOP),10)

# Database statistics
# Usage: make stats
stats:
	./venv/bin/python -m src.cli stats
