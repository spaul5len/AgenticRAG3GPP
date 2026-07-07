"""Meeting-document timeline agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rag import llm
from rag.retriever import deduplicate_results, format_evidence, hybrid_search


def build_topic_timeline(topic: str) -> str:
    """Build a chronological topic timeline from meeting documents only."""

    cleaned_topic = topic.strip()
    if not cleaned_topic:
        return "Evidence is insufficient because the topic is empty."

    results = hybrid_search(
        cleaned_topic,
        search_specs=False,
        search_meetings=True,
        k_vector=12,
        k_keyword=12,
    )
    results = _sort_results_chronologically(deduplicate_results(results))
    if not results:
        return (
            "Evidence is insufficient to build a timeline. "
            "No matching meeting documents were retrieved."
        )

    evidence = format_evidence(results)
    return llm.call_local_llm(
        _timeline_prompt(cleaned_topic, evidence),
        system_prompt=_timeline_system_prompt(),
    ).strip()


def _sort_results_chronologically(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dated: list[tuple[datetime, dict[str, Any]]] = []
    undated: list[dict[str, Any]] = []
    for result in results:
        metadata = result.get("metadata") or {}
        parsed = _parse_date(metadata.get("meeting_date"))
        if parsed is None:
            undated.append(result)
        else:
            dated.append((parsed, result))
    return [result for _date, result in sorted(dated, key=lambda item: item[0])] + undated


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def _timeline_system_prompt() -> str:
    return """
You are a source-aware 3GPP SA3 meeting timeline assistant.
Use only the supplied meeting evidence.
Return a chronological timeline when meeting_date values are available.
Do not invent dates, meetings, TDoc IDs, companies, statuses, or conclusions.
If dates are missing, group those items under "Undated evidence".
Meeting proposals remain proposals/discussions unless evidence explicitly says approved or agreed.
Every timeline item must cite evidence as [Evidence X].
""".strip()


def _timeline_prompt(topic: str, evidence: str) -> str:
    return f"""
Topic:
{topic}

Meeting evidence:
{evidence}

Build a concise chronological timeline.

Rules:
- Use meeting_date values from the evidence for chronology.
- Do not invent dates, meetings, or TDoc IDs.
- Preserve status, meeting_id, tdoc_id, source_company, and related_spec when available.
- Clearly separate proposed, agreed/approved, open, and inferred items.
- Mark model inference explicitly.
- Cite evidence as [Evidence X].
""".strip()
