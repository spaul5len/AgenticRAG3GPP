from rag import config
from rag import retriever


def result(text, metadata, **extra):
    return {
        "text": text,
        "metadata": metadata,
        "source": metadata.get("file_path", ""),
        "page": metadata.get("page"),
        **extra,
    }


def test_status_weight_orders_official_above_meeting_proposals():
    assert retriever.status_weight("official") > retriever.status_weight("approved")
    assert retriever.status_weight("approved") > retriever.status_weight("proposed")
    assert retriever.status_weight("proposed") > retriever.status_weight("unknown")


def test_sort_by_source_quality_prefers_official_for_requirement_questions():
    results = [
        result(
            "proposal text",
            {
                "doc_type": "meeting_doc",
                "status": "proposed",
                "title": "Proposal",
                "file_path": "/meeting.txt",
                "page": 1,
            },
            keyword_score=20.0,
            retrieval_source="keyword",
            query_intent={"official_requirement": True, "discussion": False},
        ),
        result(
            "official text",
            {
                "doc_type": "official_spec",
                "status": "official",
                "title": "TS 33.501",
                "file_path": "/spec.txt",
                "page": 4,
            },
            keyword_score=1.0,
            retrieval_source="keyword",
            query_intent={"official_requirement": True, "discussion": False},
        ),
    ]

    sorted_results = retriever.sort_by_source_quality(results)

    assert sorted_results[0]["metadata"]["doc_type"] == "official_spec"


def test_sort_by_source_quality_allows_meeting_docs_for_discussion_queries():
    results = [
        result(
            "official text",
            {
                "doc_type": "official_spec",
                "status": "official",
                "title": "TS 33.501",
                "file_path": "/spec.txt",
            },
            keyword_score=1.0,
            retrieval_source="keyword",
            query_intent={"official_requirement": False, "discussion": True},
        ),
        result(
            "agreed discussion",
            {
                "doc_type": "meeting_doc",
                "status": "agreed",
                "title": "Agreed TDoc",
                "file_path": "/tdoc.txt",
                "meeting_date": "2026-01-15",
            },
            keyword_score=20.0,
            retrieval_source="keyword",
            query_intent={"official_requirement": False, "discussion": True},
        ),
    ]

    sorted_results = retriever.sort_by_source_quality(results)

    assert sorted_results[0]["metadata"]["doc_type"] == "meeting_doc"


def test_deduplicate_results_merges_vector_and_keyword_scores():
    results = [
        result(
            "same text",
            {"file_path": "/spec.txt", "page": 1, "chunk_id": "chunk-1"},
            retrieval_source="vector",
            distance=0.4,
        ),
        result(
            "same text",
            {"file_path": "/spec.txt", "page": 1, "chunk_id": "chunk-1"},
            retrieval_source="keyword",
            keyword_score=3.0,
        ),
    ]

    deduped = retriever.deduplicate_results(results)

    assert len(deduped) == 1
    assert deduped[0]["retrieval_source"] == "keyword+vector"
    assert deduped[0]["distance"] == 0.4
    assert deduped[0]["keyword_score"] == 3.0


def test_hybrid_search_uses_selected_collections_and_filters_keyword(monkeypatch):
    vector_calls = []

    def fake_search_collection(collection_name, query, k, where=None):
        vector_calls.append(collection_name)
        return [
            result(
                "official vector",
                {
                    "doc_type": "official_spec",
                    "status": "official",
                    "collection_name": collection_name,
                    "file_path": "/spec.txt",
                    "page": 1,
                },
                distance=0.2,
            )
        ]

    class FakeBM25:
        def search(self, query, k):
            return [
                result(
                    "meeting keyword",
                    {
                        "doc_type": "meeting_doc",
                        "status": "proposed",
                        "collection_name": config.MEETING_COLLECTION,
                        "file_path": "/tdoc.txt",
                    },
                    retrieval_source="keyword",
                    keyword_score=9.0,
                ),
                result(
                    "spec keyword",
                    {
                        "doc_type": "official_spec",
                        "status": "official",
                        "collection_name": config.SPEC_COLLECTION,
                        "file_path": "/spec.txt",
                    },
                    retrieval_source="keyword",
                    keyword_score=2.0,
                ),
            ]

    monkeypatch.setattr(retriever, "search_collection", fake_search_collection)
    monkeypatch.setattr(retriever, "build_default_bm25", lambda: FakeBM25())

    results = retriever.hybrid_search(
        "what shall UE do", search_specs=True, search_meetings=False
    )

    assert vector_calls == [config.SPEC_COLLECTION]
    assert all(item["metadata"]["doc_type"] == "official_spec" for item in results)


def test_expand_3gpp_query_adds_acronym_specific_terms():
    expanded = retriever.expand_3gpp_query(
        "What does TS 33.501 say about AUSF and UDM in 5G authentication?"
    )

    assert "Nausf_UEAuthentication" in expanded
    assert "Authentication Server Function" in expanded
    assert "SEAF" in expanded
    assert "authentication vector" in expanded
    assert "Unified Data Management" in expanded
    assert "Nudm_UEAuthentication" in expanded
    assert "authentication subscription data" in expanded


def test_expand_3gpp_query_adds_sba_terms():
    expanded = retriever.expand_3gpp_query("How does SBA authorization work?")

    assert "Service-Based Architecture" in expanded
    assert "NF Service Consumer" in expanded
    assert "NF Service Producer" in expanded
    assert "NRF" in expanded
    assert "access token" in expanded


def test_hybrid_search_uses_expanded_query_for_vector_and_keyword(monkeypatch):
    vector_queries = []
    keyword_queries = []

    def fake_search_collection(collection_name, query, k, where=None):
        vector_queries.append(query)
        return []

    class FakeBM25:
        def search(self, query, k):
            keyword_queries.append(query)
            return []

    monkeypatch.setattr(retriever, "search_collection", fake_search_collection)
    monkeypatch.setattr(retriever, "build_default_bm25", lambda: FakeBM25())

    retriever.hybrid_search(
        "AUSF and UDM authentication",
        search_specs=True,
        search_meetings=False,
    )

    assert len(vector_queries) == 1
    assert "Nausf_UEAuthentication" in vector_queries[0]
    assert "Nudm_UEAuthentication" in vector_queries[0]
    assert keyword_queries == vector_queries


def test_acronym_exact_match_reranks_above_generic_chunk(monkeypatch):
    def fake_search_collection(collection_name, query, k, where=None):
        return [
            result(
                "Generic authentication procedure text without target functions.",
                {
                    "doc_type": "official_spec",
                    "status": "official",
                    "collection_name": collection_name,
                    "file_path": "/generic.txt",
                    "page": 1,
                },
                distance=0.05,
            ),
            result(
                "The AUSF obtains data from UDM via Nudm_UEAuthentication and derives KSEAF with SEAF.",
                {
                    "doc_type": "official_spec",
                    "status": "official",
                    "collection_name": collection_name,
                    "file_path": "/specific.txt",
                    "page": 2,
                },
                distance=0.9,
            ),
        ]

    class FakeBM25:
        def search(self, query, k):
            return []

    monkeypatch.setattr(retriever, "search_collection", fake_search_collection)
    monkeypatch.setattr(retriever, "build_default_bm25", lambda: FakeBM25())

    results = retriever.hybrid_search(
        "What does TS 33.501 say about AUSF and UDM in 5G authentication?",
        search_specs=True,
        search_meetings=False,
    )

    assert results[0]["metadata"]["file_path"] == "/specific.txt"
    assert {"AUSF", "UDM", "SEAF", "Nudm_UEAuthentication", "KSEAF"}.issubset(
        set(results[0]["exact_term_matches"])
    )


def test_format_evidence_includes_required_source_fields():
    evidence = retriever.format_evidence(
        [
            result(
                "0123456789abcdef",
                {
                    "doc_type": "meeting_doc",
                    "status": "proposed",
                    "title": "TDoc title",
                    "source_company": "Example Corp",
                    "meeting_id": "SA3_123",
                    "tdoc_id": "S3-230001",
                    "meeting_date": "2026-01-15",
                    "related_spec": "TS 33.501",
                    "file_path": "/tdoc.txt",
                    "page": 7,
                },
            )
        ],
        max_chars_per_chunk=10,
    )

    assert "Evidence 1" in evidence
    assert "doc_type: meeting_doc" in evidence
    assert "status: proposed" in evidence
    assert "title: TDoc title" in evidence
    assert "source_company: Example Corp" in evidence
    assert "meeting_id: SA3_123" in evidence
    assert "tdoc_id: S3-230001" in evidence
    assert "meeting_date: 2026-01-15" in evidence
    assert "related_spec: TS 33.501" in evidence
    assert "file_path: /tdoc.txt" in evidence
    assert "page: 7" in evidence
    assert "text: 0123456..." in evidence
