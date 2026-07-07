"""Polite download helpers with checksum manifest support."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

from rag import config
from rag.sync.source_registry import USER_AGENT


MANIFEST_PATH = config.DATA_DIR / "sync_manifest.json"
DEFAULT_DELAY_SECONDS = 1.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"downloads": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def manifest_has_url(manifest: dict[str, Any], remote_url: str) -> bool:
    return any(item.get("remote_url") == remote_url for item in manifest.get("downloads", []))


def manifest_has_hash(manifest: dict[str, Any], file_hash: str) -> bool:
    return any(item.get("file_hash") == file_hash for item in manifest.get("downloads", []))


def manifest_entry_for_url(
    manifest: dict[str, Any], remote_url: str
) -> dict[str, Any] | None:
    for item in manifest.get("downloads", []):
        if item.get("remote_url") == remote_url:
            return item
    return None


def manifest_entry_for_hash(
    manifest: dict[str, Any], file_hash: str
) -> dict[str, Any] | None:
    for item in manifest.get("downloads", []):
        if item.get("file_hash") == file_hash:
            return item
    return None


def record_download(
    manifest: dict[str, Any],
    *,
    remote_url: str,
    local_path: Path,
    file_hash: str,
    metadata: dict[str, Any],
) -> None:
    entry = {
        "remote_url": remote_url,
        "local_path": str(local_path.resolve()),
        "file_hash": file_hash,
        "downloaded_at": metadata.get("downloaded_at") or utc_now(),
        **metadata,
    }
    manifest.setdefault("downloads", [])
    manifest["downloads"] = [
        item for item in manifest["downloads"] if item.get("remote_url") != remote_url
    ]
    manifest["downloads"].append(entry)


def download_file(
    remote_url: str,
    destination_dir: Path,
    *,
    manifest: dict[str, Any] | None = None,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    timeout: int = 60,
) -> tuple[Path, str, bool]:
    """Download one file if needed and return path, sha256, downloaded flag."""

    if urlparse(remote_url).scheme != "https":
        raise ValueError(f"Only HTTPS downloads are supported: {remote_url}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = _filename_from_url(remote_url)
    destination = destination_dir / filename
    active_manifest = manifest if manifest is not None else load_manifest()

    if destination.exists():
        file_hash = sha256_file(destination)
        return destination, file_hash, False
    existing_entry = manifest_entry_for_url(active_manifest, remote_url)
    if existing_entry:
        existing_path = Path(str(existing_entry.get("local_path", "")))
        if existing_path.exists():
            return existing_path, sha256_file(existing_path), False

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    with requests.get(
        remote_url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        stream=True,
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    file_hash = sha256_file(destination)
    duplicate_entry = manifest_entry_for_hash(active_manifest, file_hash)
    if duplicate_entry:
        duplicate_path = Path(str(duplicate_entry.get("local_path", "")))
        destination.unlink(missing_ok=True)
        if duplicate_path.exists():
            return duplicate_path, file_hash, False
        return destination, file_hash, False
    return destination, file_hash, True


def write_sidecar_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _filename_from_url(remote_url: str) -> str:
    path = urlparse(remote_url).path.rstrip("/")
    filename = unquote(Path(path).name)
    if not filename:
        raise ValueError(f"Could not determine filename from URL: {remote_url}")
    return filename
