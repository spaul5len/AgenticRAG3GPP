"""Hybrid source-aware retrieval for official specs and SA3 meeting documents."""

from __future__ import annotations

import re
from typing import Any

from rag import config
from rag.keyword_index import build_default_bm25
from rag.vector_db import search_collection


OFFICIAL_QUERY_TERMS = {
    "shall",
    "must",
    "requirement",
    "requirements",
    "required",
    "specified",
    "specification",
    "standard",
    "clause",
    "normative",
}
DISCUSSION_QUERY_TERMS = {
    "proposal",
    "proposals",
    "proposed",
    "tdoc",
    "tdocs",
    "discussion",
    "discussions",
    "recent",
    "meeting",
    "minutes",
    "agenda",
    "company",
}
ACRONYM_EXPANSIONS = {
    "AUSF": [
        "Nausf_UEAuthentication",
        "Authentication Server Function",
        "SEAF",
        "authentication vector",
        "HXRES*",
        "KSEAF",
    ],
    "UDM": [
        "Unified Data Management",
        "Nudm_UEAuthentication",
        "authentication subscription data",
    ],
    "SBA": [
        "Service-Based Architecture",
        "NF Service Consumer",
        "NF Service Producer",
        "NRF",
        "access token",
    ],
}


def hybrid_search(
    query: str,
    search_specs: bool = True,
    search_meetings: bool = True,
    k_vector: int = 8,
    k_keyword: int = 8,
) -> list[dict[str, Any]]:
    """Search vector and BM25 indexes, then rank by source quality."""

    if k_vector <= 0:
        raise ValueError("k_vector must be greater than 0.")
    if k_keyword <= 0:
        raise ValueError("k_keyword must be greater than 0.")
    if not query.strip() or not (search_specs or search_meetings):
        return []

    expanded_query = expand_3gpp_query(query)
    exact_terms = exact_3gpp_terms_for_query(query)
    results: list[dict[str, Any]] = []
    query_intent = {
        "official_requirement": _is_official_requirement_query(query),
        "discussion": _is_discussion_query(query),
    }

    if search_specs:
        results.extend(
            _tag_results(
                search_collection(config.SPEC_COLLECTION, expanded_query, k=k_vector),
                retrieval_source="vector",
                query_intent=query_intent,
                exact_terms=exact_terms,
                expanded_query=expanded_query,
            )
        )
    if search_meetings:
        results.extend(
            _tag_results(
                search_collection(config.MEETING_COLLECTION, expanded_query, k=k_vector),
                retrieval_source="vector",
                query_intent=query_intent,
                exact_terms=exact_terms,
                expanded_query=expanded_query,
            )
        )

    keyword_results = build_default_bm25().search(expanded_query, k=k_keyword)
    for result in keyword_results:
        metadata = result.get("metadata") or {}
        collection_name = metadata.get("collection_name") or result.get("collection_name")
        if collection_name == config.SPEC_COLLECTION and not search_specs:
            continue
        if collection_name == config.MEETING_COLLECTION and not search_meetings:
            continue
        result["query_intent"] = query_intent
        result["exact_terms"] = exact_terms
        result["expanded_query"] = expanded_query
        results.append(result)

    return sort_by_source_quality(deduplicate_results(results))


def expand_3gpp_query(query: str) -> str:
    """Append common 3GPP acronym expansions to improve retrieval recall."""

    expanded_parts = [query.strip()]
    for acronym, expansions in ACRONYM_EXPANSIONS.items():
        if not _contains_acronym(query, acronym):
            continue
        for term in expansions:
            if term.lower() not in query.lower():
                expanded_parts.append(term)
    return " ".join(part for part in expanded_parts if part)


def exact_3gpp_terms_for_query(query: str) -> list[str]:
    """Return exact terms that should boost acronym-specific chunks."""

    terms: list[str] = []
    for acronym, expansions in ACRONYM_EXPANSIONS.items():
        if not _contains_acronym(query, acronym):
            continue
        terms.append(acronym)
        terms.extend(expansions)
    return _dedupe_case_insensitive(terms)


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate chunks while preserving vector and keyword signals."""

    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in results:
        metadata = result.get("metadata") or {}
        key = (
            str(metadata.get("file_path") or result.get("source") or ""),
            str(metadata.get("page") or result.get("page") or ""),
            str(metadata.get("chunk_id") or result.get("id") or ""),
            _text_fingerprint(str(result.get("text") or "")),
        )
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = dict(result)
            continue

        existing_sources = set(str(existing.get("retrieval_source", "")).split("+"))
        new_sources = set(str(result.get("retrieval_source", "")).split("+"))
        existing["retrieval_source"] = "+".join(sorted(existing_sources | new_sources))

        for score_key in ("keyword_score", "distance"):
            if score_key not in result:
                continue
            if score_key == "distance":
                current = existing.get(score_key)
                if current is None or result[score_key] < current:
                    existing[score_key] = result[score_key]
            else:
                existing[score_key] = max(
                    float(existing.get(score_key, 0.0)), float(result[score_key])
                )

        existing_metadata = existing.get("metadata") or {}
        existing_metadata.update({k: v for k, v in metadata.items() if v not in (None, "")})
        existing["metadata"] = existing_metadata

    return list(deduped.values())


def status_weight(status: str | None) -> float:
    """Return a conservative source-quality weight for document status."""

    normalized = (status or "").strip().lower()
    if normalized == "official":
        return 1.0
    if normalized in {"approved", "agreed"}:
        return 0.75
    if normalized in {"minutes", "noted"}:
        return 0.55
    if normalized in {"proposed", "withdrawn"}:
        return 0.25
    if normalized in {"unknown", ""}:
        return 0.15
    return 0.35


def sort_by_source_quality(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort retrieval hits by source quality, retrieval score, and recency."""

    scored = [(result, _source_quality_score(result)) for result in results]
    for result, score in scored:
        result["source_quality_score"] = score
    return [result for result, _score in sorted(scored, key=lambda item: item[1], reverse=True)]


def format_evidence(
    results: list[dict[str, Any]], max_chars_per_chunk: int = 1500
) -> str:
    """Format retrieval results as source-preserving evidence blocks."""

    if max_chars_per_chunk <= 0:
        raise ValueError("max_chars_per_chunk must be greater than 0.")

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        text = str(result.get("text") or metadata.get("text") or "")
        if len(text) > max_chars_per_chunk:
            text = text[: max_chars_per_chunk - 3].rstrip() + "..."

        blocks.append(
            "\n".join(
                [
                    f"Evidence {index}",
                    f"doc_type: {_value(metadata.get('doc_type'))}",
                    f"status: {_value(metadata.get('status'))}",
                    f"title: {_value(metadata.get('title'))}",
                    f"source_company: {_value(metadata.get('source_company'))}",
                    f"meeting_id: {_value(metadata.get('meeting_id'))}",
                    f"tdoc_id: {_value(metadata.get('tdoc_id'))}",
                    f"meeting_date: {_value(metadata.get('meeting_date'))}",
                    f"related_spec: {_value(metadata.get('related_spec'))}",
                    f"file_path: {_value(metadata.get('file_path') or result.get('source'))}",
                    f"page: {_value(metadata.get('page') or result.get('page'))}",
                    f"text: {text}",
                ]
            )
        )

    return "\n\n".join(blocks)


def _tag_results(
    results: list[dict[str, Any]],
    retrieval_source: str,
    query_intent: dict[str, bool],
    exact_terms: list[str],
    expanded_query: str,
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for result in results:
        copy = dict(result)
        copy["retrieval_source"] = retrieval_source
        copy["query_intent"] = query_intent
        copy["exact_terms"] = exact_terms
        copy["expanded_query"] = expanded_query
        tagged.append(copy)
    return tagged


def _source_quality_score(result: dict[str, Any]) -> float:
    metadata = result.get("metadata") or {}
    doc_type = str(metadata.get("doc_type") or "")
    status = str(metadata.get("status") or "")
    query_intent = result.get("query_intent") or {}

    score = status_weight(status) * 10.0
    if doc_type == "official_spec":
        score += 6.0
        if query_intent.get("official_requirement"):
            score += 4.0
    elif doc_type == "meeting_doc":
        score += 1.5
        if query_intent.get("discussion"):
            score += 4.0
        elif status.strip().lower() in {"proposed", "unknown", ""}:
            score -= 5.0

    score += _retrieval_score(result)
    score += _exact_term_score(result)
    score += _date_score(metadata.get("meeting_date"))
    return score


def _retrieval_score(result: dict[str, Any]) -> float:
    score = 0.0
    if result.get("keyword_score") is not None:
        score += min(float(result["keyword_score"]), 20.0) / 4.0
    if result.get("distance") is not None:
        distance = max(float(result["distance"]), 0.0)
        score += 1.0 / (1.0 + distance)
    if "vector" in str(result.get("retrieval_source", "")):
        score += 0.25
    if "keyword" in str(result.get("retrieval_source", "")):
        score += 0.25
    return score


def _date_score(value: Any) -> float:
    if not value:
        return 0.0
    match = re.search(r"\b(20\d{2})[-/.]?(\d{2})?[-/.]?(\d{2})?\b", str(value))
    if not match:
        return 0.0
    year = int(match.group(1))
    return max(0.0, min((year - 2000) / 100.0, 0.5))


def _is_official_requirement_query(query: str) -> bool:
    tokens = _query_tokens(query)
    return bool(tokens & OFFICIAL_QUERY_TERMS)


def _is_discussion_query(query: str) -> bool:
    tokens = _query_tokens(query)
    return bool(tokens & DISCUSSION_QUERY_TERMS)


def _query_tokens(query: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_.-]+", query)}


def _contains_acronym(text: str, acronym: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(acronym)}(?![A-Za-z0-9_])",
        text,
        flags=re.IGNORECASE,
    ) is not None


def _exact_term_score(result: dict[str, Any]) -> float:
    terms = result.get("exact_terms") or []
    if not terms:
        result["exact_term_matches"] = []
        return 0.0

    metadata = result.get("metadata") or {}
    haystack = " ".join(
        str(value)
        for value in (
            result.get("text"),
            metadata.get("text"),
            metadata.get("title"),
            metadata.get("tdoc_id"),
            metadata.get("related_spec"),
        )
        if value
    )
    matches = [term for term in terms if _contains_exact_term(haystack, str(term))]
    result["exact_term_matches"] = matches
    return float(len(matches)) * 8.0


def _contains_exact_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    if re.fullmatch(r"[A-Za-z0-9_]+", term):
        return re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
            text,
            flags=re.IGNORECASE,
        ) is not None
    return term.lower() in text.lower()


def _dedupe_case_insensitive(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _text_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:200]


def _value(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)
