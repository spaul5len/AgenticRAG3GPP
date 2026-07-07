from rag import timeline_agent


def make_result(text, meeting_date="", meeting_id="SA3_123", tdoc_id="S3-230001"):
    return {
        "text": text,
        "metadata": {
            "doc_type": "meeting_doc",
            "status": "proposed",
            "title": text,
            "meeting_id": meeting_id,
            "tdoc_id": tdoc_id,
            "meeting_date": meeting_date,
            "file_path": f"/{tdoc_id}.txt",
            "page": 1,
        },
        "source": f"/{tdoc_id}.txt",
        "page": 1,
    }


def test_build_topic_timeline_searches_meetings_only_and_orders_by_date(monkeypatch):
    calls = {}
    def fake_hybrid_search(topic, search_specs, search_meetings, k_vector, k_keyword):
        calls["search"] = (topic, search_specs, search_meetings, k_vector, k_keyword)
        return [
            make_result("later", "2026-03-01", tdoc_id="S3-260002"),
            make_result("earlier", "2026-01-01", tdoc_id="S3-260001"),
        ]

    monkeypatch.setattr(timeline_agent, "hybrid_search", fake_hybrid_search)
    captured = {}
    monkeypatch.setattr(
        timeline_agent.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: captured.setdefault(
            "values", (prompt, system_prompt)
        )
        and "2026-01-01: earlier [Evidence 1]\n2026-03-01: later [Evidence 2]",
    )

    output = timeline_agent.build_topic_timeline("AKMA")

    assert calls["search"] == ("AKMA", False, True, 12, 12)
    prompt, system_prompt = captured["values"]
    assert prompt.find("tdoc_id: S3-260001") < prompt.find("tdoc_id: S3-260002")
    assert "Do not invent dates, meetings, or TDoc IDs" in prompt
    assert "Do not invent dates, meetings, TDoc IDs" in system_prompt
    assert "2026-01-01" in output


def test_build_topic_timeline_reports_no_evidence(monkeypatch):
    monkeypatch.setattr(timeline_agent, "hybrid_search", lambda *args, **kwargs: [])

    output = timeline_agent.build_topic_timeline("unknown")

    assert "Evidence is insufficient" in output


def test_build_topic_timeline_rejects_empty_topic():
    assert "Evidence is insufficient" in timeline_agent.build_topic_timeline("  ")
