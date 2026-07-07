#!/usr/bin/env python
"""Extract DOCX figures into local data/figures storage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import config, metadata_db
from rag.figure_extractor import extract_figures_from_file
from rag.ingest_meetings import (
    build_meeting_metadata,
    iter_metadata_files,
    locate_meeting_file,
    read_metadata_rows,
)
from rag.ingest_specs import build_spec_metadata, iter_spec_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract figures from local 3GPP DOCX specs and meeting documents."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report figure extraction targets without writing images or SQLite metadata.",
    )
    args = parser.parse_args(argv)

    if not args.dry_run:
        metadata_db.init_db()

    documents = list(_iter_docx_documents())
    extracted = 0
    failed = 0
    for path, metadata in documents:
        try:
            if args.dry_run:
                print(f"[dry-run] Would extract figures from {path}")
                continue
            figures = extract_figures_from_file(
                path,
                doc_type=str(metadata.get("doc_type") or "unknown"),
                status=str(metadata.get("status") or "unknown"),
            )
            for figure in figures:
                metadata_db.register_figure(figure)
            extracted += len(figures)
            print(f"Extracted {len(figures)} figure(s) from {path}")
        except Exception as exc:
            failed += 1
            print(f"Failed to extract figures from {path}: {exc}")

    print(
        "Figure extraction complete: "
        f"documents={len(documents)}, figures={extracted}, failed={failed}"
    )
    return 1 if failed else 0


def _iter_docx_documents() -> list[tuple[Path, dict[str, str | None]]]:
    documents: list[tuple[Path, dict[str, str | None]]] = []

    if config.SPECS_DIR.exists():
        for path in iter_spec_files(config.SPECS_DIR):
            if path.suffix.lower() == ".docx":
                documents.append((path, build_spec_metadata(path)))

    if config.MEETINGS_DIR.exists():
        for metadata_path in iter_metadata_files(config.MEETINGS_DIR):
            meeting_root = metadata_path.parent
            for row in read_metadata_rows(metadata_path):
                file_name = row.get("file_name", "").strip()
                if not file_name:
                    continue
                path = locate_meeting_file(meeting_root, file_name)
                if path and path.suffix.lower() == ".docx":
                    documents.append((path, build_meeting_metadata(row, metadata_path)))

    return sorted(documents, key=lambda item: str(item[0]))


if __name__ == "__main__":
    raise SystemExit(main())
