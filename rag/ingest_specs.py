"""Official 3GPP specification ingestion."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag import config
from rag import metadata_db
from rag.chunking import chunk_pages
from rag.parsers import parse_document
from rag.vector_db import add_chunks_with_stats


SUPPORTED_SPEC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class IngestStats:
    scanned: int = 0
    indexed: int = 0
    skipped_indexed: int = 0
    skipped_unsupported: int = 0
    skipped_empty: int = 0
    failed: int = 0
    chunks: int = 0
    chunks_total: int = 0
    chunks_failed: int = 0


def ingest_specs(specs_dir: Path | None = None, dry_run: bool = False) -> IngestStats:
    """Ingest official specs from ``data/specs`` into SQLite and Chroma."""

    root = specs_dir or config.SPECS_DIR
    stats = IngestStats()

    if not root.exists():
        print(f"Specs directory does not exist: {root}")
        return stats

    for path in iter_spec_files(root):
        stats.scanned += 1

        if path.suffix.lower() not in SUPPORTED_SPEC_EXTENSIONS:
            stats.skipped_unsupported += 1
            continue

        try:
            already_indexed = _is_already_indexed(path, dry_run=dry_run)
            if already_indexed:
                stats.skipped_indexed += 1
                print(f"Skipping already indexed spec: {path}")
                continue

            pages = parse_document(path)
            chunks = chunk_pages(
                pages,
                chunk_size=config.CHUNK_SIZE_WORDS,
                overlap=config.CHUNK_OVERLAP_WORDS,
            )
            if not chunks:
                stats.skipped_empty += 1
                print(f"Skipping empty spec: {path}")
                continue

            file_hash = metadata_db.file_hash(path)
            metadata = build_spec_metadata(path)
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
            }
            chunk_stats = add_chunks_with_stats(
                config.SPEC_COLLECTION, chunks, vector_metadata
            )
            stats.chunks_total += chunk_stats.chunks_total
            stats.chunks_failed += chunk_stats.chunks_failed
            if chunk_stats.chunks_indexed <= 0:
                stats.failed += 1
                print(
                    f"Failed to index {path}: no chunks were embedded successfully "
                    f"(chunks_total={chunk_stats.chunks_total}, "
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
            print(f"Failed to index {path}: {exc}")

    return stats


def iter_spec_files(root: Path) -> Iterable[Path]:
    """Yield files below ``root`` in stable order."""

    return sorted(path for path in root.rglob("*") if path.is_file())


def build_spec_metadata(path: Path) -> dict[str, str | None]:
    """Build official-spec metadata for SQLite and vector storage."""

    related_spec = infer_related_spec(path.name)
    return {
        "doc_type": "official_spec",
        "collection_name": config.SPEC_COLLECTION,
        "title": related_spec or path.stem,
        "source_company": None,
        "meeting_id": None,
        "tdoc_id": None,
        "agenda_item": None,
        "work_item": None,
        "release": None,
        "status": "official",
        "related_spec": related_spec,
        "meeting_date": None,
        "remote_url": None,
        "source_type": "local_file",
        "downloaded_at": None,
    }


def infer_related_spec(filename: str) -> str | None:
    """Infer specs like ``TS 33.501`` from common local filenames."""

    stem = Path(filename).stem
    patterns = (
        r"(?i)\bTS[\s_.-]*33[\s_.-]*(\d{3})\b",
        r"(?i)\b33[\s_.-]*(\d{3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return f"TS 33.{match.group(1)}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index official 3GPP specifications.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan, parse, and chunk specs without writing SQLite, Chroma, or embeddings.",
    )
    args = parser.parse_args(argv)

    stats = ingest_specs(dry_run=args.dry_run)
    print(
        "Spec ingestion complete: "
        f"scanned={stats.scanned}, indexed={stats.indexed}, chunks={stats.chunks}, "
        f"chunks_total={stats.chunks_total}, chunks_indexed={stats.chunks}, "
        f"chunks_failed={stats.chunks_failed}, "
        f"skipped_indexed={stats.skipped_indexed}, "
        f"skipped_unsupported={stats.skipped_unsupported}, "
        f"skipped_empty={stats.skipped_empty}, failed={stats.failed}"
    )
    return 1 if stats.failed else 0


def _is_already_indexed(path: Path, dry_run: bool) -> bool:
    if dry_run and not Path(config.SQLITE_PATH).exists():
        return False
    return metadata_db.is_already_indexed(path)


if __name__ == "__main__":
    raise SystemExit(main())
