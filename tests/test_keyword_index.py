from rag import config
from rag import keyword_index


class FakeCollection:
    def __init__(self, ids, documents, metadatas):
        self._ids = ids
        self._documents = documents
        self._metadatas = metadatas

    def get(self, include=None):
        return {
            "ids": self._ids,
            "documents": self._documents,
            "metadatas": self._metadatas,
        }


def test_build_default_bm25_loads_specs_and_meetings_from_chroma(monkeypatch):
    collections = {
        config.SPEC_COLLECTION: FakeCollection(
            ["spec-1"],
            ["The UE shall support authentication."],
            [
                {
                    "doc_type": "official_spec",
                    "status": "official",
                    "collection_name": config.SPEC_COLLECTION,
                    "file_path": "/spec.txt",
                }
            ],
        ),
        config.MEETING_COLLECTION: FakeCollection(
            ["meeting-1"],
            ["A company proposed authentication changes."],
            [
                {
                    "doc_type": "meeting_doc",
                    "status": "proposed",
                    "collection_name": config.MEETING_COLLECTION,
                    "file_path": "/tdoc.txt",
                }
            ],
        ),
    }
    monkeypatch.setattr(keyword_index, "get_collection", lambda name: collections[name])

    index = keyword_index.build_default_bm25()
    results = index.search("shall proposed", k=10)

    assert len(index.documents) == 2
    assert {result["metadata"]["doc_type"] for result in results} == {
        "official_spec",
        "meeting_doc",
    }
    assert all(result["retrieval_source"] == "keyword" for result in results)


def test_bm25_index_rejects_invalid_k():
    index = keyword_index.BM25Index([])

    try:
        index.search("query", k=0)
    except ValueError as exc:
        assert "k must be greater than 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_tokenize_keeps_3gpp_style_tokens():
    assert keyword_index.tokenize("TS 33.501 SA3-123") == ["ts", "33.501", "sa3-123"]
