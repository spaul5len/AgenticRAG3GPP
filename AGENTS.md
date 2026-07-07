# Project: Local 3GPP SA3 Agentic RAG

## Goal

Build a local agentic RAG system for 3GPP SA3/security research.

The system must support:
1. Official 3GPP specification ingestion.
2. 3GPP meeting/TDoc ingestion.
3. Source-aware retrieval.
4. Gap analysis.
5. SA3-style contribution drafting.
6. Draft verification.
7. ChatGPT-like Streamlit UI.
8. Optional FastAPI backend.
9. Optional 3GPP public file-server sync.

## Runtime architecture

The runtime system must use:

- Ollama for local chat LLM
- Ollama embedding model
- Chroma for local vector DB in MVP
- SQLite for metadata
- BM25 for keyword search
- Python for agent logic
- Streamlit for first GUI
- FastAPI for API backend

Do not require OpenAI API for the runtime RAG system.

Codex is only used to build and debug the code.

## Important data rules

Official specs and meeting documents must be handled separately.

Official specs:
- doc_type = official_spec
- status = official

Meeting documents:
- doc_type = meeting_doc
- status can be proposed, noted, agreed, approved, withdrawn, minutes, unknown
- never treat meeting documents as approved requirements unless metadata explicitly says approved/agreed

Generated summaries:
- doc_type = generated_summary
- status = generated_summary

## Hard rules

- Do not invent clause numbers.
- Do not invent TDoc IDs.
- Do not invent company names.
- Do not treat meeting proposals as approved standard text.
- Always preserve source metadata.
- Every answer should be able to show evidence.
- Drafting must separate:
  - official specification facts
  - meeting discussions
  - identified gaps
  - proposed new text
  - model inference
- Sensitive files under data/ must not be committed.
- Do not remove .gitignore protections.
- Keep the project usable on a normal laptop.

## Desired folder structure

rag/
  config.py
  llm.py
  parsers.py
  chunking.py
  metadata_db.py
  vector_db.py
  ingest_specs.py
  ingest_meetings.py
  keyword_index.py
  retriever.py
  router.py
  pipeline.py
  gap_agent.py
  drafting_agent.py
  verifier_agent.py
  timeline_agent.py
  meeting_summary.py
  sync/
    __init__.py
    source_registry.py
    http_listing.py
    downloader.py
    spec_sync.py
    meeting_sync.py
    scheduler.py

scripts/
  index_specs.py
  index_meetings.py
  sync_3gpp_specs.py
  sync_3gpp_meetings.py
  sync_all_3gpp.py

tests/
  test_chunking.py
  test_metadata_db.py
  test_router.py

app_streamlit.py
app_fastapi.py

## Commands

Create environment:

python -m venv .venv

Install dependencies:

pip install -r requirements.txt

Index official specs:

python scripts/index_specs.py

Index meeting documents:

python scripts/index_meetings.py

Run Streamlit UI:

streamlit run app_streamlit.py

Run FastAPI backend:

uvicorn app_fastapi:app --reload

Run tests:

pytest

## Done means

A task is done only when:
1. Code is implemented.
2. Imports work.
3. Basic tests pass.
4. Runtime uses local Ollama.
5. Data remains ignored by git.
6. Source metadata is preserved.
7. Official specs and meeting docs remain clearly separated.
8. The user can run the described command successfully.