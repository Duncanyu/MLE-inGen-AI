from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "data"

CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
FAISS_INDEX = DATA_DIR / "faiss_index.faiss"
FAISS_METAS = DATA_DIR / "faiss_metas.pkl"

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CHUNKS_DIR = DATA_DIR
