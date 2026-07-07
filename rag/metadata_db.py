"""SQLite metadata storage for indexed source documents."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag import config


DOCUMENT_COLUMNS = (
    "file_path",
    "file_hash",
    "doc_type",
    "collection_name",
    "title",
    "source_company",
    "meeting_id",
    "tdoc_id",
    "agenda_item",
    "work_item",
    "release",
    "status",
    "related_spec",
    "meeting_date",
    "remote_url",
    "source_type",
    "downloaded_at",
    "indexed_at",
)
FIGURE_COLUMNS = (
    "document_id",
    "figure_number",
    "caption",
    "image_path",
    "source_file",
    "page",
    "clause",
    "surrounding_text",
    "doc_type",
    "status",
    "created_at",
)


def connect() -> sqlite3.Connection:
    """Open a connection to the configured SQLite metadata database."""

    db_path = Path(config.SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    """Create metadata tables if they do not already exist."""

    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                doc_type TEXT,
                collection_name TEXT,
                title TEXT,
                source_company TEXT,
                meeting_id TEXT,
                tdoc_id TEXT,
                agenda_item TEXT,
                work_item TEXT,
                release TEXT,
                status TEXT,
                related_spec TEXT,
                meeting_date TEXT,
                remote_url TEXT,
                source_type TEXT,
                downloaded_at TEXT,
                indexed_at TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS figures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                figure_number TEXT,
                caption TEXT,
                image_path TEXT NOT NULL UNIQUE,
                source_file TEXT NOT NULL,
                page TEXT,
                clause TEXT,
                surrounding_text TEXT,
                doc_type TEXT,
                status TEXT,
                created_at TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_figures_document_id ON figures(document_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_figures_source_file ON figures(source_file)"
        )


def file_hash(path: str | Path) -> str:
    """Return the SHA-256 hash for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_already_indexed(path: str | Path) -> bool:
    """Return True when the current file content has already been registered."""

    target_path = str(Path(path).resolve())
    target_hash = file_hash(path)
    init_db()
    with connect() as connection:
        row = connection.execute(
            """
            SELECT id FROM documents
            WHERE file_path = ? AND file_hash = ?
            LIMIT 1
            """,
            (target_path, target_hash),
        ).fetchone()
    return row is not None


def register_document(path: str | Path, metadata: dict[str, Any]) -> int:
    """Insert or update a document metadata row and return its ID."""

    init_db()
    target_path = str(Path(path).resolve())
    target_hash = file_hash(path)
    now = _utc_now()

    values = {column: metadata.get(column) for column in DOCUMENT_COLUMNS}
    values["file_path"] = target_path
    values["file_hash"] = target_hash
    values["indexed_at"] = metadata.get("indexed_at") or now

    assignments = ", ".join(
        f"{column} = excluded.{column}" for column in DOCUMENT_COLUMNS if column != "file_path"
    )
    placeholders = ", ".join("?" for _ in DOCUMENT_COLUMNS)
    columns = ", ".join(DOCUMENT_COLUMNS)
    params = tuple(values[column] for column in DOCUMENT_COLUMNS)

    with connect() as connection:
        cursor = connection.execute(
            f"""
            INSERT INTO documents ({columns})
            VALUES ({placeholders})
            ON CONFLICT(file_path) DO UPDATE SET {assignments}
            """,
            params,
        )
        row = connection.execute(
            "SELECT id FROM documents WHERE file_path = ?", (target_path,)
        ).fetchone()

    if row is None:
        raise RuntimeError(f"Failed to register document: {target_path}")
    return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])


def get_document_id(path: str | Path) -> int | None:
    """Return the registered document ID for a path, if available."""

    init_db()
    target_path = str(Path(path).resolve())
    with connect() as connection:
        row = connection.execute(
            "SELECT id FROM documents WHERE file_path = ? LIMIT 1", (target_path,)
        ).fetchone()
    if row is None:
        return None
    return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])


def register_figure(metadata: dict[str, Any]) -> int:
    """Insert or update figure metadata and return its ID."""

    init_db()
    values = {column: metadata.get(column) for column in FIGURE_COLUMNS}
    values["image_path"] = str(Path(str(values["image_path"])).resolve())
    values["source_file"] = str(Path(str(values["source_file"])).resolve())
    values["created_at"] = metadata.get("created_at") or _utc_now()

    assignments = ", ".join(
        f"{column} = excluded.{column}" for column in FIGURE_COLUMNS if column != "image_path"
    )
    placeholders = ", ".join("?" for _ in FIGURE_COLUMNS)
    columns = ", ".join(FIGURE_COLUMNS)
    params = tuple(values[column] for column in FIGURE_COLUMNS)

    with connect() as connection:
        connection.execute(
            f"""
            INSERT INTO figures ({columns})
            VALUES ({placeholders})
            ON CONFLICT(image_path) DO UPDATE SET {assignments}
            """,
            params,
        )
        row = connection.execute(
            "SELECT id FROM figures WHERE image_path = ?", (values["image_path"],)
        ).fetchone()

    if row is None:
        raise RuntimeError(f"Failed to register figure: {values['image_path']}")
    return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])


def list_figures(limit: int = 100) -> list[dict[str, Any]]:
    """Return recently registered figure metadata rows."""

    if limit <= 0:
        raise ValueError("limit must be greater than 0.")
    init_db()
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                document_id,
                figure_number,
                caption,
                image_path,
                source_file,
                page,
                clause,
                surrounding_text,
                doc_type,
                status,
                created_at
            FROM figures
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_documents(limit: int = 100) -> list[dict[str, Any]]:
    """Return recently indexed document metadata rows."""

    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    init_db()
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                file_path,
                file_hash,
                doc_type,
                collection_name,
                title,
                source_company,
                meeting_id,
                tdoc_id,
                agenda_item,
                work_item,
                release,
                status,
                related_spec,
                meeting_date,
                remote_url,
                source_type,
                downloaded_at,
                indexed_at
            FROM documents
            ORDER BY indexed_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
