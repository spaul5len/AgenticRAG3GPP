"""Project configuration for the local 3GPP SA3 Agentic RAG system."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
SPECS_DIR = DATA_DIR / "specs"
MEETINGS_DIR = DATA_DIR / "meetings"
CHROMA_DIR = BASE_DIR / "chroma_db"
SQLITE_PATH = BASE_DIR / "metadata.sqlite"

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:1.5b"
#CHAT_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

SPEC_COLLECTION = "official_3gpp_specs"
MEETING_COLLECTION = "sa3_meeting_documents"
FIGURE_COLLECTION = "document_figures"

CHUNK_SIZE_WORDS = 800
CHUNK_OVERLAP_WORDS = 120
