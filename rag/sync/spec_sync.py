"""Sync selected 3GPP specification archives."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from rag import config
from rag.sync.downloader import (
    download_file,
    load_manifest,
    record_download,
    save_manifest,
    sha256_file,
    utc_now,
    write_sidecar_metadata,
)
from rag.sync.http_listing import DirectoryLink, fetch_directory_listing, parse_directory_links
from rag.sync.source_registry import SPEC_SOURCES, SpecSource


SPEC_ARCHIVE_EXTENSIONS = {".zip"}
SPEC_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md"}


def sync_specs(sources: list[SpecSource] | None = None) -> list[dict[str, Any]]:
    """Download and unpack selected 3GPP spec archives conservatively."""

    manifest = load_manifest()
    synced: list[dict[str, Any]] = []
    for source in sources or SPEC_SOURCES:
        html = fetch_directory_listing(source.directory_url)
        links = parse_directory_links(html, source.directory_url)
        archive_links = [link for link in links if _is_spec_archive_link(link)]
        for link in archive_links:
            version = extract_spec_version(link.url)
            spec_dir = config.SPECS_DIR / _spec_dir_name(source.spec_number)
            archive_path, file_hash, downloaded = download_file(
                link.url,
                spec_dir / "archives",
                manifest=manifest,
            )
            if not file_hash and archive_path.exists():
                file_hash = sha256_file(archive_path)
            metadata = {
                "remote_url": link.url,
                "file_hash": file_hash,
                "downloaded_at": utc_now(),
                "spec_number": source.spec_number,
                "version": version,
                "source_type": "3gpp_public_file_server",
            }
            write_sidecar_metadata(archive_path, metadata)
            extracted = unzip_spec_archive(archive_path, spec_dir, metadata)
            record_download(
                manifest,
                remote_url=link.url,
                local_path=archive_path,
                file_hash=file_hash,
                metadata=metadata,
            )
            synced.append(
                {
                    **metadata,
                    "local_path": str(archive_path),
                    "downloaded": downloaded,
                    "extracted_files": [str(path) for path in extracted],
                }
            )
    save_manifest(manifest)
    return synced


def unzip_spec_archive(
    archive_path: Path, spec_dir: Path, metadata: dict[str, Any]
) -> list[Path]:
    """Unzip spec documents from an archive into the spec directory."""

    if archive_path.suffix.lower() != ".zip" or not archive_path.exists():
        return []

    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_name = Path(member.filename).name
            if not member_name or member.is_dir():
                continue
            if Path(member_name).suffix.lower() not in SPEC_DOCUMENT_EXTENSIONS:
                continue
            destination = spec_dir / member_name
            if destination.exists():
                extracted.append(destination)
                continue
            spec_dir.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
            write_sidecar_metadata(destination, metadata)
            extracted.append(destination)
    return extracted


def extract_spec_version(value: str) -> str:
    """Extract 3GPP archive version from filenames such as 33501-h10.zip."""

    filename = Path(value).name
    match = re.search(r"(?i)(?:^|[^0-9])33\d{3}[-_]?([a-z]\d{2})", filename)
    if match:
        return match.group(1).lower()
    match = re.search(r"(?i)[-_]v?(\d+\.\d+\.\d+)", filename)
    if match:
        return match.group(1)
    return ""


def _is_spec_archive_link(link: DirectoryLink) -> bool:
    return Path(link.url).suffix.lower() in SPEC_ARCHIVE_EXTENSIONS


def _spec_dir_name(spec_number: str) -> str:
    return spec_number.replace(" ", "_").replace(".", "_")
