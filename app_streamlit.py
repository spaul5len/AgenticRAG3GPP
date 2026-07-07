"""Streamlit chat UI for the local 3GPP SA3 Agentic RAG system."""

from __future__ import annotations

from typing import Any

import streamlit as st

from rag import config
from rag.drafting_agent import draft_sa3_contribution
from rag.figure_retriever import search_figures
from rag.gap_agent import analyze_gap
from rag.pipeline import answer_question
from rag.timeline_agent import build_topic_timeline
from rag.verifier_agent import draft_verify_revise


MODES = ("Ask", "Gap Analysis", "Draft Contribution", "Timeline")


def main() -> None:
    st.set_page_config(page_title="Local SA3 Agentic RAG", layout="wide")
    _ensure_session_state()
    options = render_sidebar()
    render_chat(options)


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption("Chat history")
        st.info("Current chat")

        mode = st.radio("Mode", MODES, index=0)

        st.divider()
        st.caption("Source filters")
        search_specs = st.checkbox("Official Specs", value=True)
        search_meetings = st.checkbox("Meeting Docs", value=True)
        st.checkbox("Internal Docs", value=False, disabled=True)

        st.divider()
        st.caption("Advanced options")
        max_chunks = st.slider("Max chunks", min_value=1, max_value=30, value=8)
        show_evidence = st.checkbox("Show evidence", value=True)
        retrieve_figures = st.checkbox("Retrieve figures", value=True)
        run_verifier = st.checkbox("Run verifier for drafts", value=True)

        st.divider()
        st.caption("Local status")
        st.status(f"Ollama: {config.OLLAMA_URL}", state="complete")
        st.status(f"Embedding model: {config.EMBED_MODEL}", state="complete")
        st.status(f"Vector DB: {config.CHROMA_DIR}", state="complete")

    return {
        "mode": mode,
        "search_specs": search_specs,
        "search_meetings": search_meetings,
        "max_chunks": max_chunks,
        "show_evidence": show_evidence,
        "retrieve_figures": retrieve_figures,
        "run_verifier": run_verifier,
    }


def render_chat(options: dict[str, Any]) -> None:
    st.title("Local SA3 Agentic RAG")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            _render_message_content(message, options)

    prompt = st.chat_input(_input_placeholder(options["mode"]))
    if not prompt:
        return

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Working locally..."):
            assistant_message = run_mode(prompt, options)
        _render_message_content(assistant_message, options)

    st.session_state.messages.append(assistant_message)


def run_mode(prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    mode = options["mode"]
    try:
        if mode == "Ask":
            retrieve_figures = bool(options.get("retrieve_figures", True))
            content = _call_answer_question(
                prompt,
                retrieve_figures=retrieve_figures,
            )
            figures = _safe_search_figures(prompt, k=3) if retrieve_figures else []
            message = {
                "role": "assistant",
                "content": content,
                "kind": "answer",
            }
            if figures:
                message["package"] = {"figures": figures}
            return message
        if mode == "Gap Analysis":
            package = analyze_gap(prompt)
            return {
                "role": "assistant",
                "content": str(package.get("gap_analysis", "")),
                "kind": "gap",
                "package": package,
            }
        if mode == "Draft Contribution":
            if options.get("run_verifier", True):
                package = draft_verify_revise(prompt)
            else:
                draft_package = draft_sa3_contribution(prompt)
                package = {
                    "topic": draft_package.get("topic", prompt.strip()),
                    "draft_type": draft_package.get("draft_type", "discussion contribution"),
                    "draft": draft_package.get("draft", ""),
                    "review": "",
                    "revised_draft": draft_package.get("draft", ""),
                    "gap_package": draft_package.get("gap_package", {}),
                }
            return {
                "role": "assistant",
                "content": str(package.get("revised_draft") or package.get("draft") or ""),
                "kind": "draft",
                "package": package,
            }
        if mode == "Timeline":
            content = build_topic_timeline(prompt)
            return {"role": "assistant", "content": content, "kind": "timeline"}
    except Exception as exc:
        return {
            "role": "assistant",
            "content": f"Error: {_safe_error_message(exc)}",
            "kind": "error",
        }

    return {"role": "assistant", "content": "Unsupported mode.", "kind": "error"}


def _call_answer_question(prompt: str, retrieve_figures: bool) -> str:
    try:
        return answer_question(
            prompt,
            include_figures=retrieve_figures,
            max_figures=3,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return answer_question(prompt)


def _render_message_content(message: dict[str, Any], options: dict[str, Any]) -> None:
    if message.get("kind") == "draft":
        _render_draft_message(message, options)
    else:
        st.markdown(str(message.get("content", "")))
        if options.get("show_evidence", True):
            _render_evidence_panel(message)


def _render_draft_message(message: dict[str, Any], options: dict[str, Any]) -> None:
    package = message.get("package") or {}
    gap_package = package.get("gap_package") or {}
    revised_tab, initial_tab, review_tab, evidence_tab = st.tabs(
        ["Revised Draft", "Initial Draft", "Verifier Review", "Evidence"]
    )

    with revised_tab:
        st.markdown(str(package.get("revised_draft") or package.get("draft") or ""))
    with initial_tab:
        st.markdown(str(package.get("draft") or ""))
    with review_tab:
        review = str(package.get("review") or "")
        if review and options.get("run_verifier", True):
            st.warning(review)
        else:
            st.info("Verifier was not run for this draft.")
    with evidence_tab:
        st.markdown(str(gap_package.get("evidence") or "No evidence returned."))

    if options.get("show_evidence", True):
        _render_evidence_panel(message)


def _render_evidence_panel(message: dict[str, Any]) -> None:
    package = message.get("package") or {}
    evidence = package.get("evidence") or (package.get("gap_package") or {}).get("evidence")
    raw_results = package.get("raw_results") or (package.get("gap_package") or {}).get(
        "raw_results"
    )
    figures = package.get("figures") or []
    review = package.get("review")

    if not evidence and not raw_results and not figures and not review:
        return

    with st.expander("Evidence", expanded=False):
        if review:
            st.warning(str(review))
        if figures:
            for index, figure in enumerate(figures, start=1):
                _render_figure_card(index, figure)
        if raw_results:
            for index, result in enumerate(raw_results, start=1):
                _render_source_card(index, result)
        elif evidence:
            st.markdown(str(evidence))


def _render_source_card(index: int, result: dict[str, Any]) -> None:
    metadata = result.get("metadata") or {}
    title = metadata.get("title") or metadata.get("file_path") or f"Source {index}"
    with st.container(border=True):
        st.markdown(f"**Evidence {index}: {title}**")
        st.caption(str(result.get("text", ""))[:500])
        st.json(
            {
                "doc_type": metadata.get("doc_type"),
                "status": metadata.get("status"),
                "source_company": metadata.get("source_company"),
                "meeting_id": metadata.get("meeting_id"),
                "tdoc_id": metadata.get("tdoc_id"),
                "meeting_date": metadata.get("meeting_date"),
                "related_spec": metadata.get("related_spec"),
                "file_path": metadata.get("file_path") or result.get("source"),
                "page": metadata.get("page") or result.get("page"),
            }
        )


def _render_figure_card(index: int, figure: dict[str, Any]) -> None:
    caption = figure.get("caption") or "Extracted figure"
    image_path = str(figure.get("image_path") or "")
    with st.container(border=True):
        st.markdown(f"**Figure Evidence {index}: {caption}**")
        if image_path:
            try:
                st.image(image_path, caption=caption, use_container_width=True)
            except TypeError:
                st.image(image_path, caption=caption)
            except Exception as exc:
                st.warning(f"Could not display image: {_safe_error_message(exc)}")
        st.json(
            {
                "figure_number": figure.get("figure_number"),
                "source_file": figure.get("source_file"),
                "page_or_order": figure.get("page")
                or (figure.get("metadata") or {}).get("approximate_order"),
                "clause": figure.get("clause"),
                "doc_type": figure.get("doc_type"),
                "status": figure.get("status"),
            }
        )


def _ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _input_placeholder(mode: str) -> str:
    if mode == "Ask":
        return "Ask about SA3 specs, TDocs, or security topics"
    if mode == "Gap Analysis":
        return "Enter a topic for gap analysis"
    if mode == "Draft Contribution":
        return "Enter a topic for an SA3 contribution draft"
    return "Enter a topic for timeline analysis"


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    parts = []
    for token in message.split():
        normalized = token.replace("\\", "/")
        if "/data/" in normalized or normalized.startswith("data/"):
            parts.append("[redacted-data-path]")
        else:
            parts.append(token)
    return " ".join(parts)


def _safe_search_figures(prompt: str, k: int) -> list[dict[str, Any]]:
    try:
        return search_figures(prompt, k=k)
    except Exception:
        return []


if __name__ == "__main__":
    main()
