"""Meeting summarization and generated-summary indexing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag import config, llm, metadata_db
from rag.chunking import chunk_text
from rag.retriever import deduplicate_results, format_evidence, hybrid_search
from rag.vector_db import add_chunks


SUMMARY_DIR = config.DATA_DIR / "generated_summaries"


def summarize_meeting(meeting_id: str) -> str:
    """Summarize one meeting while separating proposals, agreements, issues, and inference."""

    cleaned_meeting_id = meeting_id.strip()
    if not cleaned_meeting_id:
        return "Evidence is insufficient because meeting_id is empty."

    results = hybrid_search(
        cleaned_meeting_id,
        search_specs=False,
        search_meetings=True,
        k_vector=20,
        k_keyword=20,
    )
    results = [
        result
        for result in deduplicate_results(results)
        if str((result.get("metadata") or {}).get("meeting_id") or "").strip()
        == cleaned_meeting_id
    ]
    if not results:
        return (
            f"Evidence is insufficient to summarize meeting {cleaned_meeting_id}. "
            "No matching meeting documents were retrieved."
        )

    evidence = format_evidence(results)
    return llm.call_local_llm(
        _summary_prompt(cleaned_meeting_id, evidence),
        system_prompt=_summary_system_prompt(),
    ).strip()


def index_meeting_summary(meeting_id: str, meeting_date: str = "") -> str:
    """Generate, store, and index a meeting summary as generated-summary evidence."""

    cleaned_meeting_id = meeting_id.strip()
    if not cleaned_meeting_id:
        return "Evidence is insufficient because meeting_id is empty."

    summary = summarize_meeting(cleaned_meeting_id)
    summary_path = _summary_path(cleaned_meeting_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")

    metadata = {
        "doc_type": "meeting_summary",
        "collection_name": config.MEETING_COLLECTION,
        "title": f"Generated summary for {cleaned_meeting_id}",
        "source_company": None,
        "meeting_id": cleaned_meeting_id,
        "tdoc_id": None,
        "agenda_item": None,
        "work_item": None,
        "release": None,
        "status": "generated_summary",
        "related_spec": None,
        "meeting_date": meeting_date.strip() or None,
        "remote_url": None,
        "source_type": "generated_summary",
        "downloaded_at": None,
    }
    chunks = chunk_text(summary, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
    vector_metadata = {
        **metadata,
        "doc_id": f"meeting-summary-{cleaned_meeting_id}",
        "file_path": str(summary_path.resolve()),
        "source": str(summary_path.resolve()),
    }
    add_chunks(config.MEETING_COLLECTION, chunks, vector_metadata)
    metadata_db.register_document(summary_path, metadata)
    return summary


def _summary_path(meeting_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", meeting_id).strip("_") or "meeting"
    return SUMMARY_DIR / f"{safe_id}_summary.md"


def _summary_system_prompt() -> str:
    return """
You are a source-aware 3GPP SA3 meeting summarization assistant.
Use only the supplied evidence.
The generated summary must separate proposals, agreed items, open issues, and inference.
Do not invent dates, meetings, TDoc IDs, companies, positions, statuses, or decisions.
Do not treat proposed or unknown-status meeting documents as approved.
Mark model inference explicitly.
Every factual statement must cite evidence as [Evidence X].
""".strip()


def _summary_prompt(meeting_id: str, evidence: str) -> str:
    return f"""
Meeting ID:
{meeting_id}

Evidence:
{evidence}

Generate a meeting summary with exactly these sections:
1. Proposals
2. Agreed items
3. Open issues
4. Inference
5. Evidence map

Rules:
- Preserve meeting document status.
- Do not treat proposals as approved.
- Mark model inference explicitly.
- Cite evidence as [Evidence X].
""".strip()
