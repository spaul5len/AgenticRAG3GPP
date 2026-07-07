"""3GPP SA3 meeting/TDoc ingestion."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag import config
from rag import metadata_db
from rag.chunking import chunk_pages
from rag.parsers import parse_document
from rag.vector_db import add_chunks_with_stats


SUPPORTED_MEETING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md"}
METADATA_COLUMNS = (
    "file_name",
    "tdoc_id",
    "title",
    "source_company",
    "agenda_item",
    "work_item",
    "release",
    "status",
    "related_spec",
    "meeting_id",
    "meeting_date",
)


@dataclass
class IngestStats:
    metadata_files: int = 0
    scanned: int = 0
    indexed: int = 0
    skipped_missing: int = 0
    skipped_indexed: int = 0
    skipped_unsupported: int = 0
    skipped_empty: int = 0
    failed: int = 0
    chunks: int = 0
    chunks_total: int = 0
    chunks_failed: int = 0


def ingest_meetings(
    meetings_dir: Path | None = None, dry_run: bool = False
) -> IngestStats:
    """Ingest SA3 meeting documents described by meeting-level metadata CSVs."""

    root = meetings_dir or config.MEETINGS_DIR
    stats = IngestStats()

    if not root.exists():
        print(f"Meetings directory does not exist: {root}")
        return stats

    for metadata_path in iter_metadata_files(root):
        stats.metadata_files += 1
        meeting_root = metadata_path.parent
        for row_number, row in enumerate(read_metadata_rows(metadata_path), start=2):
            stats.scanned += 1
            file_name = row.get("file_name", "").strip()
            if not file_name:
                stats.skipped_missing += 1
                print(f"Skipping metadata row without file_name: {metadata_path}:{row_number}")
                continue

            path = locate_meeting_file(meeting_root, file_name)
            if path is None:
                stats.skipped_missing += 1
                print(f"Skipping missing meeting document: {meeting_root / file_name}")
                continue

            if path.suffix.lower() not in SUPPORTED_MEETING_EXTENSIONS:
                stats.skipped_unsupported += 1
                continue

            try:
                already_indexed = _is_already_indexed(path, dry_run=dry_run)
                if already_indexed:
                    stats.skipped_indexed += 1
                    print(f"Skipping already indexed meeting document: {path}")
                    continue

                pages = parse_document(path)
                chunks = chunk_pages(
                    pages,
                    chunk_size=config.CHUNK_SIZE_WORDS,
                    overlap=config.CHUNK_OVERLAP_WORDS,
                )
                if not chunks:
                    stats.skipped_empty += 1
                    print(f"Skipping empty meeting document: {path}")
                    continue

                file_hash = metadata_db.file_hash(path)
                metadata = build_meeting_metadata(row, metadata_path)
                if dry_run:
                    stats.indexed += 1
                    stats.chunks += len(chunks)
                    stats.chunks_total += len(chunks)
                    print(f"[dry-run] Would index {path} ({len(chunks)} chunks)")
                    continue

                vector_metadata = {
                    **metadata,
                    "doc_id": file_hash,
                    "file_hash": file_hash,
                    "file_path": str(path.resolve()),
                    "source": str(path.resolve()),
                    "metadata_csv": str(metadata_path.resolve()),
                }
                chunk_stats = add_chunks_with_stats(
                    config.MEETING_COLLECTION, chunks, vector_metadata
                )
                stats.chunks_total += chunk_stats.chunks_total
                stats.chunks_failed += chunk_stats.chunks_failed
                if chunk_stats.chunks_indexed <= 0:
                    stats.failed += 1
                    print(
                        f"Failed to index meeting document {path}: no chunks were "
                        f"embedded successfully (chunks_total={chunk_stats.chunks_total}, "
                        f"chunks_failed={chunk_stats.chunks_failed})"
                    )
                    continue

                metadata_db.register_document(path, metadata)
                stats.indexed += 1
                stats.chunks += chunk_stats.chunks_indexed
                print(
                    f"Indexed {path} "
                    f"(chunks_total={chunk_stats.chunks_total}, "
                    f"chunks_indexed={chunk_stats.chunks_indexed}, "
                    f"chunks_failed={chunk_stats.chunks_failed})"
                )
            except Exception as exc:
                stats.failed += 1
                print(f"Failed to index meeting document {path}: {exc}")

    return stats


def iter_metadata_files(root: Path) -> Iterable[Path]:
    """Yield meeting metadata CSV files in stable order."""

    return sorted(path for path in root.rglob("metadata.csv") if path.is_file())


def read_metadata_rows(metadata_path: Path) -> list[dict[str, str]]:
    """Read a meeting metadata CSV into normalized string dictionaries."""

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [
            {
                column: (row.get(column) or "").strip()
                for column in METADATA_COLUMNS
            }
            for row in reader
        ]


def locate_meeting_file(meeting_root: Path, file_name: str) -> Path | None:
    """Find a meeting document in the meeting root, tdocs, or minutes folder."""

    source = Path(file_name)
    candidates = (
        meeting_root / source,
        meeting_root / "tdocs" / source,
        meeting_root / "minutes" / source,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_meeting_metadata(
    row: dict[str, str], metadata_path: Path | None = None
) -> dict[str, str | None]:
    """Build metadata for a meeting document without upgrading it to official."""

    title = _clean_value(row.get("title")) or _clean_value(row.get("file_name"))
    meeting_id = _clean_value(row.get("meeting_id"))
    if meeting_id is None and metadata_path is not None:
        meeting_id = metadata_path.parent.name

    return {
        "doc_type": "meeting_doc",
        "collection_name": config.MEETING_COLLECTION,
        "title": title,
        "source_company": _clean_value(row.get("source_company")),
        "meeting_id": meeting_id,
        "tdoc_id": _clean_value(row.get("tdoc_id")),
        "agenda_item": _clean_value(row.get("agenda_item")),
        "work_item": _clean_value(row.get("work_item")),
        "release": _clean_value(row.get("release")),
        "status": _clean_value(row.get("status")) or "unknown",
        "related_spec": _clean_value(row.get("related_spec")),
        "meeting_date": _clean_value(row.get("meeting_date")),
        "remote_url": None,
        "source_type": "local_file",
        "downloaded_at": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index 3GPP SA3 meeting documents.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan, parse, and chunk meeting documents without writing SQLite, Chroma, or embeddings.",
    )
    args = parser.parse_args(argv)

    stats = ingest_meetings(dry_run=args.dry_run)
    print(
        "Meeting ingestion complete: "
        f"metadata_files={stats.metadata_files}, scanned={stats.scanned}, "
        f"indexed={stats.indexed}, chunks={stats.chunks}, "
        f"chunks_total={stats.chunks_total}, chunks_indexed={stats.chunks}, "
        f"chunks_failed={stats.chunks_failed}, "
        f"skipped_missing={stats.skipped_missing}, "
        f"skipped_indexed={stats.skipped_indexed}, "
        f"skipped_unsupported={stats.skipped_unsupported}, "
        f"skipped_empty={stats.skipped_empty}, failed={stats.failed}"
    )
    return 1 if stats.failed else 0


def _is_already_indexed(path: Path, dry_run: bool) -> bool:
    if dry_run and not Path(config.SQLITE_PATH).exists():
        return False
    return metadata_db.is_already_indexed(path)


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
