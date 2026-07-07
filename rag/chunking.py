"""Word-based chunking helpers that preserve source page metadata."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict[str, str | int]]:
    """Split text into word chunks with deterministic chunk IDs."""

    _validate_chunk_settings(chunk_size, overlap)
    words = text.split()
    if not words:
        return []

    chunks: list[dict[str, str | int]] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(
                {
                    "chunk_id": f"chunk-{chunk_index}",
                    "text": " ".join(chunk_words),
                    "start_word": start,
                    "end_word": end,
                }
            )
            chunk_index += 1
        if end == len(words):
            break
        start += step

    return chunks


def chunk_pages(
    pages: list[dict[str, str | int]], chunk_size: int, overlap: int
) -> list[dict[str, str | int]]:
    """Split page records into chunks while preserving page numbers."""

    _validate_chunk_settings(chunk_size, overlap)
    chunks: list[dict[str, str | int]] = []
    chunk_index = 0

    for page in pages:
        text = str(page.get("text", ""))
        if not text.strip():
            continue
        page_number = page.get("page")
        for page_chunk in chunk_text(text, chunk_size, overlap):
            chunks.append(
                {
                    **page_chunk,
                    "chunk_id": f"chunk-{chunk_index}",
                    "page": page_number,
                }
            )
            chunk_index += 1

    return chunks


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")
