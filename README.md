# Local 3GPP SA3 Agentic RAG

## Project Goal

Build a local agentic RAG system for 3GPP SA3 and security research. The system is intended to ingest official 3GPP specifications and meeting/TDoc material, preserve source metadata, retrieve evidence, analyze gaps, draft SA3-style contributions, verify drafts, and provide a ChatGPT-like local UI.

Official specifications and meeting documents must remain clearly separated. Meeting proposals must not be treated as approved requirements unless their metadata explicitly says they are approved or agreed.

## Local Runtime Stack

The runtime system is local-first:

- Ollama for the chat LLM
- Ollama embedding model
- Chroma for the local vector database
- SQLite for metadata
- BM25 for keyword search
- Python for agent logic
- Streamlit for the first GUI
- FastAPI for an optional API backend

The runtime RAG system does not require the OpenAI API.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Indexing

Index official specifications:

```bash
python scripts/index_specs.py
```

Index meeting documents:

```bash
python scripts/index_meetings.py
```

## Streamlit UI

Run the local chat UI:

```bash
streamlit run app_streamlit.py
```

## Local Data Warning

The `data/` directory is ignored by git and should contain local documents only. Put downloaded 3GPP specifications, meeting documents, TDocs, and other sensitive or large source files under `data/`, not in tracked source files.

Do not remove the `.gitignore` protections for local documents, databases, generated indexes, or virtual environments.
