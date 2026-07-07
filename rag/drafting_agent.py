"""SA3-style contribution drafting agent."""

from __future__ import annotations

from typing import Any

from rag import llm
from rag.gap_agent import analyze_gap


def draft_sa3_contribution(
    topic: str, draft_type: str = "discussion contribution"
) -> dict[str, Any]:
    """Draft a cautious SA3-style contribution from a gap-analysis package."""

    gap_package = analyze_gap(topic)
    cleaned_topic = topic.strip()
    cleaned_draft_type = draft_type.strip() or "discussion contribution"

    draft = llm.call_local_llm(
        _draft_prompt(cleaned_topic, cleaned_draft_type, gap_package),
        system_prompt=_draft_system_prompt(),
    ).strip()
    if not draft:
        draft = (
            "Draft could not be generated because the local model returned an empty "
            "response. Review the gap package and evidence before drafting manually."
        )

    return {
        "topic": cleaned_topic,
        "draft_type": cleaned_draft_type,
        "draft": draft,
        "gap_package": gap_package,
    }


def _draft_system_prompt() -> str:
    return """
You are a cautious 3GPP SA3 contribution drafting assistant.
Use only the supplied gap package and evidence.
Every technical claim must cite evidence as [Evidence X].
Do not invent clause numbers.
Do not invent TDoc IDs.
Do not invent company names.
Do not invent requirements, meeting IDs, dates, or statuses.
Do not treat meeting documents as approved standard text.
Approved or agreed meeting documents remain meeting_doc evidence, not official specifications.
Use cautious SA3-style language, including:
- SA3 is invited to discuss
- It is proposed to study
- The following aspects may be considered
Use "shall" only for proposed normative text or verified official text explicitly supported by evidence.
Clearly label weak claims and missing evidence.
""".strip()


def _draft_prompt(topic: str, draft_type: str, gap_package: dict[str, Any]) -> str:
    return f"""
Topic:
{topic}

Draft type:
{draft_type}

Gap analysis:
{gap_package.get("gap_analysis", "")}

Evidence:
{gap_package.get("evidence", "")}

Draft a cautious SA3-style {draft_type} with exactly these sections:
1. Title
2. Abstract
3. Background
4. Problem statement
5. Discussion
6. Proposal
7. Questions for SA3
8. Possible normative direction
9. Evidence map
10. Weak claims / missing evidence

Rules:
- Do not invent clause numbers.
- Do not invent TDoc IDs.
- Do not invent companies.
- Do not treat meeting docs as approved standard text.
- Use cautious language such as "SA3 is invited to discuss", "It is proposed to study", and "The following aspects may be considered".
- Use "shall" only for proposed normative text or verified official text.
- Cite evidence as [Evidence X].
""".strip()
