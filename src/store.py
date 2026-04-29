"""SQLite-backed document store with numpy vector matrix.

Documents are stored in SQLite with FTS5 full-text search.
Embeddings live in a memory-mapped .npy file (6617 × 384 = ~10 MB).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE,          -- original ID from source
    title       TEXT,
    content     TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}',     -- JSON blob for extra fields
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    title,
    content,
    content=docs,
    content_rowid=id
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
    INSERT INTO docs_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
END;

CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
    INSERT INTO docs_fts(rowid, title, content)
    VALUES (new.id, new.title, new.content);
END;
"""

EMBEDDING_DIM = 384


class DocStore:
    """Manages documents in SQLite and embeddings in a .npy file."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "docs.db"
        self.vec_path = self.data_dir / "embeddings.npy"

        self._conn: sqlite3.Connection | None = None
        self._vectors: np.ndarray | None = None

    # ── connection management ──────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── vector matrix ──────────────────────────────────────────────

    @property
    def vectors(self) -> np.ndarray:
        """Return (N, 384) embedding matrix, loading on first access."""
        if self._vectors is None:
            if self.vec_path.exists():
                self._vectors = np.load(self.vec_path, mmap_mode="r")
            else:
                self._vectors = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return self._vectors

    def reload_vectors(self):
        """Force reload vectors from disk (after ingestion)."""
        self._vectors = None
        return self.vectors

    def save_vectors(self, vecs: np.ndarray):
        """Persist embedding matrix to disk."""
        np.save(self.vec_path, vecs.astype(np.float32))
        self.reload_vectors()

    # ── document operations ────────────────────────────────────────

    def add_documents(
        self,
        contents: list[str],
        embeddings: np.ndarray,
        external_ids: list[str] | None = None,
        titles: list[str] | None = None,
        metadata_list: list[dict] | None = None,
    ) -> list[int]:
        """Insert documents and append their embeddings. Returns row IDs."""
        n = len(contents)
        ids: list[int] = []

        cur = self.conn.cursor()
        for i in range(n):
            ext_id = external_ids[i] if external_ids else None
            title = titles[i] if titles else None
            meta = json.dumps(metadata_list[i]) if metadata_list else "{}"

            cur.execute(
                "INSERT INTO docs (external_id, title, content, metadata) VALUES (?, ?, ?, ?)",
                (ext_id, title, contents[i], meta),
            )
            ids.append(cur.lastrowid)
        self.conn.commit()

        # Append embeddings to matrix
        existing = (
            np.load(self.vec_path, mmap_mode="r")
            if self.vec_path.exists()
            else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        )
        combined = np.vstack([existing, embeddings]) if len(existing) > 0 else embeddings
        self.save_vectors(combined)

        return ids

    def get_document(self, row_id: int) -> dict[str, Any] | None:
        """Retrieve a single document by its internal ID."""
        row = self.conn.execute(
            "SELECT id, external_id, title, content, metadata, created_at FROM docs WHERE id = ?",
            (row_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    def get_documents(self, row_ids: list[int]) -> list[dict[str, Any]]:
        """Retrieve multiple documents by internal IDs, preserving order."""
        placeholders = ",".join("?" * len(row_ids))
        rows = self.conn.execute(
            f"SELECT id, external_id, title, content, metadata, created_at FROM docs WHERE id IN ({placeholders})",
            row_ids,
        ).fetchall()
        row_map = {r["id"]: r for r in rows}
        return [_row_to_dict(row_map[rid]) for rid in row_ids if rid in row_map]

    def keyword_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """FTS5 keyword search. Returns matching documents."""
        rows = self.conn.execute(
            """
            SELECT d.id, d.external_id, d.title, d.content, d.metadata, d.created_at,
                   rank
            FROM docs_fts f
            JOIN docs d ON f.rowid = d.id
            WHERE docs_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row)
            d["_rank"] = row["rank"]
            results.append(d)
        return results

    def count(self) -> int:
        """Total number of documents."""
        return self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def clear(self):
        """Delete all documents and embeddings."""
        self.conn.execute("DELETE FROM docs")
        self.conn.commit()
        if self.vec_path.exists():
            self.vec_path.unlink()
        self.reload_vectors()

    def __len__(self) -> int:
        return self.count()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    # Parse metadata JSON
    if "metadata" in d and isinstance(d["metadata"], str):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
    return d
