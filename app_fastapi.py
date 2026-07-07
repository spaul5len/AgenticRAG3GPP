"""FastAPI backend for the local SA3 Agentic RAG system."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.drafting_agent import draft_sa3_contribution
from rag.gap_agent import analyze_gap
from rag.pipeline import answer_question
from rag.timeline_agent import build_topic_timeline


app = FastAPI(title="Local 3GPP SA3 Agentic RAG")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    draft_type: str = "discussion contribution"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    return _handle(
        lambda: {
            "question": request.question,
            "answer": answer_question(request.question),
        }
    )


@app.post("/gap")
def gap(request: TopicRequest) -> dict[str, Any]:
    return _handle(lambda: analyze_gap(request.topic))


@app.post("/draft")
def draft(request: TopicRequest) -> dict[str, Any]:
    return _handle(
        lambda: draft_sa3_contribution(
            request.topic,
            draft_type=request.draft_type,
        )
    )


@app.post("/timeline")
def timeline(request: TopicRequest) -> dict[str, Any]:
    return _handle(
        lambda: {
            "topic": request.topic,
            "timeline": build_topic_timeline(request.topic),
        }
    )


def _handle(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        result = action()
    except Exception as exc:
        return {
            "ok": False,
            "error": _safe_error_message(exc),
        }
    return {"ok": True, "data": result}


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return _redact_sensitive_paths(message)


def _redact_sensitive_paths(message: str) -> str:
    parts = []
    for token in message.split():
        normalized = token.replace("\\", "/")
        if "/data/" in normalized or normalized.startswith("data/"):
            parts.append("[redacted-data-path]")
        else:
            parts.append(token)
    return " ".join(parts)
