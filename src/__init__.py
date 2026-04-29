"""
Heavenletters Semantic Search Engine

Usage:
    # CLI
    python -m src.cli search "creating community" --top 5
    python -m src.cli stats

    # Ingestion
    python -m src.ingest data/batch_*.json

    # API server
    uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

__version__ = "1.0.0"
