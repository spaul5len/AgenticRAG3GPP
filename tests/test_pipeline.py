from rag import pipeline


def make_result(text, metadata):
    return {
        "text": text,
        "metadata": metadata,
        "source": metadata.get("file_path", ""),
        "page": metadata.get("page"),
    }


def test_answer_question_routes_and_searches_suggested_queries(monkeypatch):
    route = {
        "search_specs": True,
        "search_meetings": True,
        "intent": "ask",
        "reason": "test route",
        "suggested_queries": ["query one", "query two"],
    }
    calls = []

    monkeypatch.setattr(pipeline, "route_query", lambda question: route)

    def fake_hybrid_search(query, search_specs=True, search_meetings=True):
        calls.append((query, search_specs, search_meetings))
        return [
            make_result(
                f"text for {query}",
                {
                    "doc_type": "official_spec",
                    "status": "official",
                    "title": "TS 33.501",
                    "file_path": f"/{query}.txt",
                    "page": 1,
                },
            )
        ]

    captured = {}

    def fake_call_local_llm(prompt, system_prompt=None):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return (
            "Official specification facts: The evidence says so [Evidence 1].\n\n"
            "Meeting discussions: No meeting evidence was retrieved."
        )

    monkeypatch.setattr(pipeline, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(pipeline.llm, "call_local_llm", fake_call_local_llm)

    answer = pipeline.answer_question("question")

    assert calls == [("query one", True, True), ("query two", True, True)]
    assert "[Evidence 1]" in answer
    assert "Official specification facts" in captured["prompt"]
    assert "Meeting discussions" in captured["prompt"]
    assert "Never treat proposed" in captured["system_prompt"]
    assert "Do not invent clause numbers" in captured["system_prompt"]
    assert "SBA means Service-Based Architecture" in captured["system_prompt"]
    assert "NF = Network Function" in captured["system_prompt"]
    assert "NRF = Network Repository Function" in captured["system_prompt"]
    assert "NF Service Consumer / Producer are SBA entities" in captured["system_prompt"]
    assert "SEPP = Security Edge Protection Proxy" in captured["system_prompt"]
    assert "AUSF = Authentication Server Function" in captured["system_prompt"]
    assert "UDM = Unified Data Management" in captured["system_prompt"]
    assert "SEAF = Security Anchor Function" in captured["system_prompt"]
    assert "Do not expand acronyms incorrectly" in captured["system_prompt"]
    assert "say so instead of guessing" in captured["system_prompt"]


def test_answer_question_formats_specs_and_meeting_evidence(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "route_query",
        lambda question: {
            "search_specs": True,
            "search_meetings": True,
            "intent": "gap",
            "reason": "gap route",
            "suggested_queries": ["gap query"],
        },
    )
    monkeypatch.setattr(
        pipeline,
        "hybrid_search",
        lambda *args, **kwargs: [
            make_result(
                "official text",
                {
                    "doc_type": "official_spec",
                    "status": "official",
                    "title": "TS 33.501",
                    "related_spec": "TS 33.501",
                    "file_path": "/spec.txt",
                    "page": 5,
                },
            ),
            make_result(
                "proposal text",
                {
                    "doc_type": "meeting_doc",
                    "status": "proposed",
                    "title": "Proposal",
                    "source_company": "Example Corp",
                    "meeting_id": "SA3_123",
                    "tdoc_id": "S3-230001",
                    "meeting_date": "2026-01-15",
                    "related_spec": "TS 33.501",
                    "file_path": "/tdoc.txt",
                    "page": 2,
                },
            ),
        ],
    )

    captured = {}
    monkeypatch.setattr(
        pipeline.llm,
        "call_local_llm",
        lambda prompt, system_prompt=None: captured.setdefault("prompt", prompt)
        or "answer [Evidence 1]",
    )

    pipeline.answer_question("Find the gap")

    assert "doc_type: official_spec" in captured["prompt"]
    assert "doc_type: meeting_doc" in captured["prompt"]
    assert "status: proposed" in captured["prompt"]
    assert "tdoc_id: S3-230001" in captured["prompt"]


def test_answer_question_reports_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "route_query",
        lambda question: {
            "search_specs": True,
            "search_meetings": True,
            "intent": "ask",
            "reason": "test",
            "suggested_queries": ["nothing"],
        },
    )
    monkeypatch.setattr(pipeline, "hybrid_search", lambda *args, **kwargs: [])

    answer = pipeline.answer_question("unknown thing")

    assert "Evidence is insufficient" in answer


def test_answer_question_rejects_empty_question():
    assert "Evidence is insufficient" in pipeline.answer_question("   ")
