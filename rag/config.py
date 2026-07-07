"""Project configuration for the local 3GPP SA3 Agentic RAG system."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
SPECS_DIR = DATA_DIR / "specs"
MEETINGS_DIR = DATA_DIR / "meetings"
CHROMA_DIR = BASE_DIR / "chroma_db"
SQLITE_PATH = BASE_DIR / "metadata.sqlite"

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

SPEC_COLLECTION = "official_specs"
MEETING_COLLECTION = "meeting_docs"

CHUNK_SIZE_WORDS = 800
CHUNK_OVERLAP_WORDS = 120
