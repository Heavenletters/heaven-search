"""SQLite-backed document store with numpy vector matrix.

Documents are stored in SQLite with FTS5 full-text search.
Embeddings live in memory-mapped .npy files — one per embedding backend.
Both Vertex (768-dim) and local bge-base (768-dim) share the same SQLite DB.
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

# Default embedding dimension — matches both Vertex (768) and bge-base (768).
# Legacy MiniLM (384) is supported via the dim parameter.
DEFAULT_EMBEDDING_DIM = 768


class DocStore:
    """Manages documents in SQLite and embeddings in a .npy file.

    Each embedding backend gets its own .npy file (e.g. embeddings_vertex.npy,
    embeddings_local.npy) sharing the same docs.db.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        dim: int = DEFAULT_EMBEDDING_DIM,
        vec_suffix: str = "vertex",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "docs.db"
        self.vec_path = self.data_dir / f"embeddings_{vec_suffix}.npy"
        self._legacy_path = self.data_dir / "embeddings.npy"

        # Auto-detect dimension from existing file (handles legacy migrations)
        self.dim = self._detect_dim(vec_suffix) or dim

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
        """Return (N, dim) embedding matrix, loading on first access.

        Checks the configured vec_path first, falls back to legacy
        embeddings.npy for migration compatibility.
        """
        if self._vectors is None:
            path = self.vec_path
            if not path.exists() and self._legacy_path.exists():
                path = self._legacy_path

            if path.exists():
                self._vectors = np.load(path, mmap_mode="r")
            else:
                self._vectors = np.empty((0, self.dim), dtype=np.float32)
        return self._vectors

    def reload_vectors(self):
        """Force reload vectors from disk (after ingestion)."""
        self._vectors = None
        return self.vectors

    def save_vectors(self, vecs: np.ndarray):
        """Persist embedding matrix to disk."""
        np.save(self.vec_path, vecs.astype(np.float32))
        self.reload_vectors()

    @property
    def has_vectors(self) -> bool:
        """Whether this backend has ingested embeddings."""
        return (
            (self.vec_path.exists() and self.vec_path.stat().st_size > 0)
            or (self._legacy_path.exists() and self._legacy_path.stat().st_size > 0)
        )

    def _detect_dim(self, vec_suffix: str = "") -> int | None:
        """Detect embedding dimension from an existing .npy file.

        Checks the configured vec_path first. For the 'local' suffix only,
        falls back to the legacy embeddings.npy (384-dim MiniLM).
        Returns None if no file exists.
        """
        paths = [self.vec_path]
        if vec_suffix == "local":
            paths.append(self._legacy_path)
        for p in paths:
            if p.exists() and p.stat().st_size > 0:
                try:
                    arr = np.load(p, mmap_mode="r")
                    return arr.shape[1]
                except (ValueError, IndexError, OSError):
                    continue
        return None

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

            # Use INSERT OR IGNORE to handle re-ingestion of same external_id
            cur.execute(
                "INSERT OR IGNORE INTO docs (external_id, title, content, metadata) VALUES (?, ?, ?, ?)",
                (ext_id, title, contents[i], meta),
            )
            ids.append(cur.lastrowid if cur.lastrowid else 0)
        self.conn.commit()

        # Always append to existing vectors — for re-ingestion, the ingest
        # script clears old vectors first, so we start fresh and accumulate.
        existing = (
            np.load(self.vec_path, mmap_mode="r")
            if self.vec_path.exists() and self.vec_path.stat().st_size > 0
            else np.empty((0, self.dim), dtype=np.float32)
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
        if not row_ids:
            return []
        placeholders = ",".join("?" * len(row_ids))
        rows = self.conn.execute(
            f"SELECT id, external_id, title, content, metadata, created_at FROM docs WHERE id IN ({placeholders})",
            row_ids,
        ).fetchall()
        row_map = {r["id"]: r for r in rows}
        return [_row_to_dict(row_map[rid]) for rid in row_ids if rid in row_map]

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Escape FTS5-breaking characters so arbitrary text works.

        FTS5 has its own mini query language. Characters like * " ( )
        and trailing sentence punctuation can cause syntax errors.

        If the query contains any dangerous characters, we wrap it in
        double quotes (literal phrase). Otherwise we pass it through
        as-is, preserving implicit-AND word matching for short queries
        like "divine love" or "finding peace".
        """
        stripped = query.strip()

        # Characters that will break FTS5 query parsing
        fts5_syntax_chars = {'*', '"', '(', ')'}

        # Trailing sentence punctuation also breaks parsing
        has_trailing_punct = stripped and stripped[-1] in '.!?,;:'

        has_dangerous = any(c in query for c in fts5_syntax_chars)

        if not has_dangerous and not has_trailing_punct:
            return query

        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def keyword_search(
        self, query: str, limit: int = 20, force_phrase: bool = False
    ) -> list[dict[str, Any]]:
        """FTS5 keyword search. Returns matching documents."""
        if force_phrase:
            escaped = query.replace('"', '""')
            safe_query = f'"{escaped}"'
        else:
            safe_query = self._sanitize_fts5_query(query)
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
            (safe_query, limit),
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

    def clear_vectors(self):
        """Delete embedding vectors only (keep documents)."""
        if self.vec_path.exists():
            self.vec_path.unlink()
        self.reload_vectors()

    def clear(self):
        """Delete all documents and embeddings."""
        self.conn.execute("DELETE FROM docs")
        self.conn.commit()
        self.clear_vectors()

    def __len__(self) -> int:
        return self.count()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if "metadata" in d and isinstance(d["metadata"], str):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
    return d
