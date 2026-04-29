.PHONY: install test dev build up down shell ingest password

# Install dependencies in a virtual environment
install:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "✓ Installation complete."
	@echo "  The first run will download all-MiniLM-L6-v2 (~80MB)."

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

# Ingest JSON batches
# Usage: make ingest FILES="data/batch_001.json data/batch_002.json"
ingest:
	./venv/bin/python -m src.ingest $(FILES)

# Generate a password hash for the admin user
password:
	@read -p "Admin password: " PASS; \
	./venv/bin/python scripts/hash_password.py "$$PASS"

# CLI search shortcut
# Usage: make query Q="creating community" TOP=5
query:
	./venv/bin/python -m src.cli search "$(Q)" --top $(or $(TOP),10)
