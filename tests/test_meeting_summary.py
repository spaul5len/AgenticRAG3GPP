from rag import meeting_summary


def make_result(text, status, meeting_id="SA3_123", tdoc_id="S3-230001"):
    return {
        "text": text,
        "metadata": {
            "doc_type": "meeting_doc",
            "status": status,
            "title": text,
            "meeting_id": meeting_id,
            "tdoc_id": tdoc_id,
            "meeting_date": "2026-01-15",
            "source_company": "Example Corp",
            "file_path": f"/{tdoc_id}.txt",
            "page": 1,
        },
        "source": f"/{tdoc_id}.txt",
        "page": 1,
    }


def test_summarize_meeting_filters_to_meeting_id_and_uses_required_sections(monkeypatch):
    monkeypatch.setattr(
        meeting_summary,
        "hybrid_search",
        lambda meeting_id, search_specs, search_meetings, k_vector, k_keyword: [
            make_result("proposal", "proposed", meeting_id="SA3_123", tdoc_id="S3-230001"),
            make_result("other", "agreed", meeting_id="SA3_124", tdoc_id="S3-240001"),
        ],
    )
    captured = {}
    monkeypatch.setattr(
        meeting_summary.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: captured.setdefault(
            "values", (prompt, system_prompt)
        )
        and "\n".join(
            [
                "1. Proposals: proposal [Evidence 1]",
                "2. Agreed items: none found",
                "3. Open issues: unclear",
                "4. Inference: Model inference: needs review",
                "5. Evidence map: Evidence 1 proposed",
            ]
        ),
    )

    summary = meeting_summary.summarize_meeting("SA3_123")

    prompt, system_prompt = captured["values"]
    assert "1. Proposals" in prompt
    assert "2. Agreed items" in prompt
    assert "3. Open issues" in prompt
    assert "4. Inference" in prompt
    assert "doc_type: meeting_doc" in prompt
    assert "meeting_id: SA3_123" in prompt
    assert "SA3_124" not in prompt
    assert "separate proposals, agreed items, open issues, and inference" in system_prompt
    assert "Proposals" in summary


def test_summarize_meeting_reports_no_evidence(monkeypatch):
    monkeypatch.setattr(meeting_summary, "hybrid_search", lambda *args, **kwargs: [])

    output = meeting_summary.summarize_meeting("SA3_999")

    assert "Evidence is insufficient" in output


def test_index_meeting_summary_writes_generated_summary_metadata(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(meeting_summary, "SUMMARY_DIR", tmp_path / "summaries")
    monkeypatch.setattr(meeting_summary, "summarize_meeting", lambda meeting_id: "summary text")

    def fake_add_chunks(collection_name, chunks, metadata):
        calls["add_chunks"] = (collection_name, chunks, metadata)
        return len(chunks)

    def fake_register_document(path, metadata):
        calls["register_document"] = (path, metadata)
        return 1

    monkeypatch.setattr(meeting_summary, "add_chunks", fake_add_chunks)
    monkeypatch.setattr(meeting_summary.metadata_db, "register_document", fake_register_document)

    summary = meeting_summary.index_meeting_summary("SA3_123", meeting_date="2026-01-15")

    assert summary == "summary text"
    collection_name, chunks, vector_metadata = calls["add_chunks"]
    assert collection_name == meeting_summary.config.MEETING_COLLECTION
    assert chunks[0]["text"] == "summary text"
    assert vector_metadata["doc_type"] == "meeting_summary"
    assert vector_metadata["status"] == "generated_summary"
    assert vector_metadata["meeting_id"] == "SA3_123"
    assert vector_metadata["meeting_date"] == "2026-01-15"
    path, sqlite_metadata = calls["register_document"]
    assert path.name == "SA3_123_summary.md"
    assert path.read_text(encoding="utf-8") == "summary text"
    assert sqlite_metadata["doc_type"] == "meeting_summary"
    assert sqlite_metadata["status"] == "generated_summary"


def test_index_meeting_summary_rejects_empty_meeting_id():
    assert "Evidence is insufficient" in meeting_summary.index_meeting_summary("   ")
