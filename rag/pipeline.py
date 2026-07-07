"""End-to-end question answering pipeline for local SA3 RAG."""

from __future__ import annotations

from typing import Any

from rag import llm
from rag.figure_retriever import format_figure_evidence, search_figures
from rag.retriever import (
    deduplicate_results,
    format_evidence,
    hybrid_search,
    sort_by_source_quality,
)
from rag.router import route_query


def answer_question(
    question: str,
    include_figures: bool = True,
    max_figures: int = 3,
) -> str:
    """Route, retrieve evidence, and answer with explicit evidence references."""

    if not question.strip():
        return "Evidence is insufficient to answer because the question is empty."

    route = route_query(question)
    suggested_queries = route.get("suggested_queries") or [question]
    results: list[dict[str, Any]] = []
    for query in suggested_queries:
        results.extend(
            hybrid_search(
                query,
                search_specs=bool(route.get("search_specs", True)),
                search_meetings=bool(route.get("search_meetings", True)),
            )
        )

    results = sort_by_source_quality(deduplicate_results(results))
    figure_results = (
        _search_figure_evidence(suggested_queries, max_figures)
        if include_figures
        else []
    )
    if not results and not figure_results:
        return (
            "Evidence is insufficient to answer this question. "
            "No matching official specification, meeting, or figure evidence was retrieved."
        )

    evidence = format_evidence(results) if results else "No text evidence retrieved."
    figure_evidence = format_figure_evidence(figure_results) if figure_results else ""
    prompt = _answer_prompt(question, route, evidence, figure_evidence)
    system_prompt = _answer_system_prompt()
    answer = llm.call_local_llm(prompt, system_prompt=system_prompt).strip()
    if not answer:
        return (
            "Evidence is insufficient to answer this question. "
            "The local model returned an empty answer."
        )
    return answer


def _answer_system_prompt() -> str:
    return """
You are a source-aware 3GPP SA3 research assistant.
Use only the supplied evidence.
Every factual claim must cite evidence as [Evidence X].
Separate official specification facts from meeting discussions.
Never treat proposed, unknown, noted, or withdrawn meeting documents as approved standard text.
Approved or agreed meeting documents remain meeting_doc evidence, not official specifications.
Do not invent clause numbers, TDoc IDs, company names, statuses, meeting IDs, dates, or requirements.
In 3GPP/5G context, SBA means Service-Based Architecture unless the evidence clearly says otherwise.
Common acronym hints:
- SBA = Service-Based Architecture
- NF = Network Function
- NRF = Network Repository Function
- NF Service Consumer / Producer are SBA entities
- SEPP = Security Edge Protection Proxy
- AUSF = Authentication Server Function
- UDM = Unified Data Management
- SEAF = Security Anchor Function
Do not expand acronyms incorrectly.
If acronym meaning is uncertain from context and evidence, say so instead of guessing.
Figure evidence can support statements about diagrams and captions, but it does not replace official text unless the evidence says so.
If the evidence is insufficient, say so plainly.
""".strip()


def _answer_prompt(
    question: str,
    route: dict[str, Any],
    evidence: str,
    figure_evidence: str = "",
) -> str:
    return f"""
Question:
{question}

Route:
intent: {route.get("intent")}
search_specs: {route.get("search_specs")}
search_meetings: {route.get("search_meetings")}
reason: {route.get("reason")}

Evidence:
{evidence}

Figure evidence:
{figure_evidence or "No figure evidence retrieved."}

Answer requirements:
- Include [Evidence X] references for claims.
- Use [Figure Evidence X] references for claims based on extracted figures.
- Use separate sections for Official specification facts and Meeting discussions when both source types appear.
- Keep proposals and unknown-status meeting documents clearly labeled as meeting discussions, not approved standards.
- State when evidence is insufficient for part of the question.
""".strip()


def _search_figure_evidence(
    queries: list[str],
    max_figures: int,
) -> list[dict[str, Any]]:
    if max_figures <= 0:
        return []

    figures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        try:
            candidates = search_figures(query, k=max_figures)
        except Exception:
            continue
        for figure in candidates:
            key = str(figure.get("image_path") or figure.get("caption") or figure)
            if key in seen:
                continue
            seen.add(key)
            figures.append(figure)
            if len(figures) >= max_figures:
                return figures
    return figures
