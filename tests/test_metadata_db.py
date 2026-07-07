import hashlib
import sqlite3

import pytest

from rag import config
from rag import metadata_db


@pytest.fixture()
def temp_sqlite_path(tmp_path, monkeypatch):
    db_path = tmp_path / "metadata.sqlite"
    monkeypatch.setattr(config, "SQLITE_PATH", db_path)
    return db_path


def test_init_db_creates_documents_table_with_required_columns(temp_sqlite_path):
    metadata_db.init_db()

    assert temp_sqlite_path.exists()
    with sqlite3.connect(temp_sqlite_path) as connection:
        rows = connection.execute("PRAGMA table_info(documents)").fetchall()

    columns = {row[1] for row in rows}
    assert {
        "id",
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
    }.issubset(columns)


def test_connect_uses_configured_temp_sqlite_path(temp_sqlite_path):
    with metadata_db.connect() as connection:
        connection.execute("CREATE TABLE smoke_test (id INTEGER)")

    with sqlite3.connect(temp_sqlite_path) as connection:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'smoke_test'"
        ).fetchone()

    assert row is not None
    assert not config.BASE_DIR.joinpath("metadata.sqlite").exists()


def test_file_hash_returns_sha256_for_file(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")

    assert metadata_db.file_hash(source) == hashlib.sha256(b"hello").hexdigest()


def test_register_document_and_list_documents(temp_sqlite_path, tmp_path):
    source = tmp_path / "spec.txt"
    source.write_text("spec text", encoding="utf-8")

    document_id = metadata_db.register_document(
        source,
        {
            "doc_type": "official_spec",
            "collection_name": "official_specs",
            "title": "TS 33.xxx",
            "release": "Rel-19",
            "status": "official",
            "source_type": "local_file",
        },
    )

    documents = metadata_db.list_documents()

    assert document_id == 1
    assert len(documents) == 1
    assert documents[0]["file_path"] == str(source.resolve())
    assert documents[0]["file_hash"] == metadata_db.file_hash(source)
    assert documents[0]["doc_type"] == "official_spec"
    assert documents[0]["collection_name"] == "official_specs"
    assert documents[0]["title"] == "TS 33.xxx"
    assert documents[0]["release"] == "Rel-19"
    assert documents[0]["status"] == "official"
    assert documents[0]["source_type"] == "local_file"
    assert documents[0]["indexed_at"]


def test_is_already_indexed_checks_path_and_current_hash(temp_sqlite_path, tmp_path):
    source = tmp_path / "meeting.txt"
    source.write_text("initial", encoding="utf-8")

    assert metadata_db.is_already_indexed(source) is False

    metadata_db.register_document(
        source,
        {
            "doc_type": "meeting_doc",
            "collection_name": "meeting_docs",
            "meeting_id": "SA3-001",
            "tdoc_id": "S3-000001",
            "status": "proposed",
        },
    )

    assert metadata_db.is_already_indexed(source) is True

    source.write_text("changed", encoding="utf-8")

    assert metadata_db.is_already_indexed(source) is False


def test_register_document_updates_existing_file_row(temp_sqlite_path, tmp_path):
    source = tmp_path / "doc.txt"
    source.write_text("text", encoding="utf-8")

    first_id = metadata_db.register_document(source, {"title": "Old"})
    second_id = metadata_db.register_document(source, {"title": "New"})

    documents = metadata_db.list_documents()
    assert first_id == second_id
    assert len(documents) == 1
    assert documents[0]["title"] == "New"


def test_list_documents_rejects_invalid_limit(temp_sqlite_path):
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        metadata_db.list_documents(limit=0)
