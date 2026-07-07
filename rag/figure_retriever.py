"""Figure retrieval helpers."""

from __future__ import annotations

from typing import Any

from rag import config
from rag.vector_db import search_collection


def search_figures(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Search indexed figure captions and surrounding text."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if not query.strip():
        return []

    results = search_collection(config.FIGURE_COLLECTION, query, k=k)
    figures: list[dict[str, Any]] = []
    for result in results:
        metadata = result.get("metadata") or {}
        figures.append(
            {
                "text": result.get("text", ""),
                "caption": metadata.get("caption", ""),
                "image_path": metadata.get("image_path", ""),
                "source_file": metadata.get("source_file", ""),
                "figure_number": metadata.get("figure_number", ""),
                "page": metadata.get("page"),
                "clause": metadata.get("clause", ""),
                "doc_type": metadata.get("doc_type", ""),
                "status": metadata.get("status", ""),
                "metadata": metadata,
                "distance": result.get("distance"),
            }
        )
    return figures


def format_figure_evidence(figures: list[dict[str, Any]]) -> str:
    """Format figure retrieval results for answer prompts."""

    blocks: list[str] = []
    for index, figure in enumerate(figures, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Figure Evidence {index}",
                    f"figure_number: {_value(figure.get('figure_number'))}",
                    f"caption: {_value(figure.get('caption'))}",
                    f"image_path: {_value(figure.get('image_path'))}",
                    f"source_file: {_value(figure.get('source_file'))}",
                    f"page_or_order: {_value(figure.get('page'))}",
                    f"clause: {_value(figure.get('clause'))}",
                    f"doc_type: {_value(figure.get('doc_type'))}",
                    f"status: {_value(figure.get('status'))}",
                    f"text: {_value(figure.get('text'))}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _value(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)
