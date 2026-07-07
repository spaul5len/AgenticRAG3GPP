"""Sync public SA3 meeting directory listings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag import config
from rag.sync.downloader import load_manifest, record_download, save_manifest, utc_now
from rag.sync.http_listing import fetch_directory_listing, parse_directory_links
from rag.sync.source_registry import MEETING_LIST_SOURCES, MeetingListSource


def sync_meeting_lists(
    sources: list[MeetingListSource] | None = None,
) -> list[dict[str, Any]]:
    """Fetch top-level meeting directory listings without recursive crawling."""

    manifest = load_manifest()
    output_dir = config.DATA_DIR / "meeting_lists"
    output_dir.mkdir(parents=True, exist_ok=True)
    synced: list[dict[str, Any]] = []

    for source in sources or MEETING_LIST_SOURCES:
        html = fetch_directory_listing(source.directory_url)
        links = parse_directory_links(html, source.directory_url)
        local_path = output_dir / f"{source.name}_listing.html"
        local_path.write_text(html, encoding="utf-8")
        metadata = {
            "remote_url": source.directory_url,
            "file_hash": _sha256_text(html),
            "downloaded_at": utc_now(),
            "source_type": "3gpp_public_file_server",
            "meeting_list_name": source.name,
            "link_count": len(links),
        }
        record_download(
            manifest,
            remote_url=source.directory_url,
            local_path=local_path,
            file_hash=metadata["file_hash"],
            metadata=metadata,
        )
        synced.append({**metadata, "local_path": str(local_path)})

    save_manifest(manifest)
    return synced


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
