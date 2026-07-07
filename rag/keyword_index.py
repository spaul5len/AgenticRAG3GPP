"""BM25 keyword indexing over locally stored Chroma documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from rag import config
from rag.vector_db import get_collection


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass
class KeywordDocument:
    """A Chroma-backed document chunk available for BM25 retrieval."""

    id: str
    text: str
    metadata: dict[str, Any]
    collection_name: str


class BM25Index:
    """Small BM25 wrapper that returns source-aware retrieval results."""

    def __init__(self, documents: list[KeywordDocument] | None = None) -> None:
        self.documents = documents or []
        self._tokenized = [tokenize(document.text) for document in self.documents]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    @classmethod
    def from_chroma_collections(cls, collection_names: list[str]) -> "BM25Index":
        """Build a BM25 index from all documents in the given Chroma collections."""

        documents: list[KeywordDocument] = []
        for collection_name in collection_names:
            documents.extend(load_collection_documents(collection_name))
        return cls(documents)

    def search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Return up to ``k`` BM25-ranked source-aware results."""

        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self._bm25 is None or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            (
                (index, float(score), len(query_token_set & set(self._tokenized[index])))
                for index, score in enumerate(scores)
            ),
            key=lambda item: (item[1], item[2]),
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for index, score, overlap in ranked:
            if score <= 0 and overlap <= 0:
                continue
            document = self.documents[index]
            metadata = dict(document.metadata)
            results.append(
                {
                    "text": document.text,
                    "metadata": metadata,
                    "source": _source_from_metadata(metadata),
                    "page": metadata.get("page"),
                    "keyword_score": float(score) if score > 0 else float(overlap) * 0.1,
                    "collection_name": document.collection_name,
                    "retrieval_source": "keyword",
                    "id": document.id,
                }
            )
            if len(results) >= k:
                break

        return results


def build_default_bm25() -> BM25Index:
    """Build BM25 over official specs and SA3 meeting documents."""

    return BM25Index.from_chroma_collections(
        [config.SPEC_COLLECTION, config.MEETING_COLLECTION]
    )


def load_collection_documents(collection_name: str) -> list[KeywordDocument]:
    """Load Chroma documents and metadata for one collection."""

    try:
        collection = get_collection(collection_name)
        raw = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    ids = raw.get("ids") or []
    texts = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    documents: list[KeywordDocument] = []
    for index, text in enumerate(texts):
        if not text or not str(text).strip():
            continue
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        document_id = ids[index] if index < len(ids) else f"{collection_name}:{index}"
        documents.append(
            KeywordDocument(
                id=str(document_id),
                text=str(text),
                metadata=dict(metadata),
                collection_name=collection_name,
            )
        )
    return documents


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 with lowercase alphanumeric/spec tokens."""

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _source_from_metadata(metadata: dict[str, Any]) -> str:
    for key in ("source", "file_path", "remote_url", "title"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""
