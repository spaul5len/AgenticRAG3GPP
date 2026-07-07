from pathlib import Path

import pytest

from rag.sync import http_listing, scheduler, spec_sync
from rag.sync import downloader
from rag.sync.downloader import record_download


def test_parse_directory_links_resolves_https_links_and_skips_parent():
    html = """
    <html><body>
      <a href="../">Parent Directory</a>
      <a href="33501-h10.zip">33501-h10.zip</a>
      <a href="subdir/">subdir/</a>
      <a href="mailto:test@example.com">mail</a>
    </body></html>
    """

    links = http_listing.parse_directory_links(
        html, "https://www.3gpp.org/ftp/Specs/archive/33_series/33.501/"
    )

    assert [link.href for link in links] == ["33501-h10.zip", "subdir/"]
    assert links[0].url == (
        "https://www.3gpp.org/ftp/Specs/archive/33_series/33.501/33501-h10.zip"
    )
    assert links[0].is_directory is False
    assert links[1].is_directory is True


def test_parse_directory_links_rejects_non_https_base_url():
    with pytest.raises(ValueError, match="Only HTTPS"):
        http_listing.parse_directory_links("<a href='x.zip'>x</a>", "http://example.com/")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("33501-h10.zip", "h10"),
        ("33501_g20.zip", "g20"),
        ("TS_33.501-v18.4.0.zip", "18.4.0"),
        ("no-version.zip", ""),
    ],
)
def test_extract_spec_version(filename, expected):
    assert spec_sync.extract_spec_version(filename) == expected


def test_unzip_spec_archive_extracts_documents_and_writes_sidecar(tmp_path):
    import zipfile

    archive_path = tmp_path / "33501-h10.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("33501-h10.docx", "doc text")
        archive.writestr("ignore.csv", "a,b")

    metadata = {
        "remote_url": "https://www.3gpp.org/ftp/Specs/archive/33_series/33.501/33501-h10.zip",
        "file_hash": "abc",
        "downloaded_at": "2026-01-01T00:00:00+00:00",
        "spec_number": "TS 33.501",
        "version": "h10",
        "source_type": "3gpp_public_file_server",
    }

    extracted = spec_sync.unzip_spec_archive(archive_path, tmp_path / "TS_33_501", metadata)

    assert [path.name for path in extracted] == ["33501-h10.docx"]
    assert extracted[0].read_bytes() == b"doc text"
    assert extracted[0].with_suffix(".docx.metadata.json").exists()


def test_record_download_replaces_existing_url_entry(tmp_path):
    manifest = {"downloads": [{"remote_url": "https://example.com/a.zip", "old": True}]}

    record_download(
        manifest,
        remote_url="https://example.com/a.zip",
        local_path=tmp_path / "a.zip",
        file_hash="hash",
        metadata={"source_type": "3gpp_public_file_server"},
    )

    assert len(manifest["downloads"]) == 1
    assert manifest["downloads"][0]["file_hash"] == "hash"
    assert manifest["downloads"][0]["source_type"] == "3gpp_public_file_server"


def test_download_file_uses_existing_manifest_url_path(tmp_path):
    existing = tmp_path / "existing.zip"
    existing.write_text("already downloaded", encoding="utf-8")
    manifest = {
        "downloads": [
            {
                "remote_url": "https://www.3gpp.org/ftp/example.zip",
                "local_path": str(existing),
                "file_hash": downloader.sha256_file(existing),
            }
        ]
    }

    path, file_hash, downloaded = downloader.download_file(
        "https://www.3gpp.org/ftp/example.zip",
        tmp_path / "downloads",
        manifest=manifest,
        delay_seconds=0,
    )

    assert path == existing
    assert file_hash == downloader.sha256_file(existing)
    assert downloaded is False


def test_create_scheduler_default_interval_is_six_hours():
    created = scheduler.create_scheduler()
    jobs = created.get_jobs()

    assert len(jobs) == 1
    assert jobs[0].trigger.interval.total_seconds() == 6 * 60 * 60


def test_create_scheduler_rejects_invalid_interval():
    with pytest.raises(ValueError, match="interval_hours"):
        scheduler.create_scheduler(interval_hours=0)
