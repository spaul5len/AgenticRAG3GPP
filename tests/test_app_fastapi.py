import pytest
from pydantic import ValidationError

import app_fastapi


def test_health():
    assert app_fastapi.health() == {"status": "ok"}


def test_routes_are_registered():
    routes = {(route.path, ",".join(sorted(route.methods))) for route in app_fastapi.app.routes}

    assert ("/health", "GET") in routes
    assert ("/ask", "POST") in routes
    assert ("/gap", "POST") in routes
    assert ("/draft", "POST") in routes
    assert ("/timeline", "POST") in routes


def test_ask_endpoint_returns_answer(monkeypatch):
    monkeypatch.setattr(app_fastapi, "answer_question", lambda question: "answer")

    response = app_fastapi.ask(app_fastapi.AskRequest(question="What is required?"))

    assert response == {
        "ok": True,
        "data": {"question": "What is required?", "answer": "answer"},
    }


def test_gap_endpoint_returns_gap_package(monkeypatch):
    monkeypatch.setattr(
        app_fastapi,
        "analyze_gap",
        lambda topic: {
            "topic": topic,
            "gap_analysis": "gap",
            "evidence": "Evidence 1",
            "raw_results": [],
        },
    )

    response = app_fastapi.gap(
        app_fastapi.TopicRequest(
            topic="AKMA",
            draft_type="discussion contribution",
        )
    )

    assert response["data"]["gap_analysis"] == "gap"


def test_draft_endpoint_passes_draft_type(monkeypatch):
    calls = {}

    def fake_draft(topic, draft_type="discussion contribution"):
        calls["args"] = (topic, draft_type)
        return {
            "topic": topic,
            "draft_type": draft_type,
            "draft": "draft",
            "gap_package": {},
        }

    monkeypatch.setattr(app_fastapi, "draft_sa3_contribution", fake_draft)

    response = app_fastapi.draft(app_fastapi.TopicRequest(topic="AKMA", draft_type="CR"))

    assert calls["args"] == ("AKMA", "CR")
    assert response["data"]["draft_type"] == "CR"


def test_timeline_endpoint_returns_timeline(monkeypatch):
    monkeypatch.setattr(
        app_fastapi,
        "build_topic_timeline",
        lambda topic: "timeline",
    )

    response = app_fastapi.timeline(app_fastapi.TopicRequest(topic="AKMA"))

    assert response == {
        "ok": True,
        "data": {"topic": "AKMA", "timeline": "timeline"},
    }


def test_endpoint_errors_are_returned_and_data_paths_are_redacted(monkeypatch):
    def fail(question):
        raise RuntimeError("failed reading data/specs/private.pdf")

    monkeypatch.setattr(app_fastapi, "answer_question", fail)

    response = app_fastapi.ask(app_fastapi.AskRequest(question="x"))

    assert response["ok"] is False
    assert "failed reading" in response["error"]
    assert "data/specs/private.pdf" not in response["error"]
    assert "[redacted-data-path]" in response["error"]


def test_validation_rejects_missing_question():
    with pytest.raises(ValidationError):
        app_fastapi.AskRequest()
