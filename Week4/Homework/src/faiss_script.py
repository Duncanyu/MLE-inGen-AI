import json, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .paths import CHUNKS_JSONL, FAISS_INDEX, FAISS_METAS

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_index = None
_metas = None
_model = None

def _l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def _ensure_loaded():
    global _index, _metas, _model
    if _index is not None and _metas is not None and _model is not None:
        return
    if not FAISS_INDEX.exists() or not FAISS_METAS.exists():
        raise FileNotFoundError("FAISS artifacts not found. Run the pipeline first.")
    _index = faiss.read_index(str(FAISS_INDEX))
    with open(FAISS_METAS, "rb") as f:
        blob = pickle.load(f)
    _metas = blob.get("metas", [])
    _model = SentenceTransformer(blob.get("model_name", MODEL_NAME))

def build_index():
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_JSONL}. Run scraping/cleaning/chunking first.")
    metas = []
    texts = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            texts.append(f"{rec.get('title','')} — {rec['text']}")
            metas.append({
                "i": i,
                "doc_id": rec.get("doc_id"),
                "title": rec.get("title"),
                "source": rec.get("source"),
                "chunk_id": rec.get("chunk_id", 0),
            })
    model = SentenceTransformer(MODEL_NAME)
    X = model.encode(texts, show_progress_bar=True).astype("float32")
    X = _l2_normalize(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX))
    with FAISS_METAS.open("wb") as f:
        pickle.dump({"metas": metas, "model_name": MODEL_NAME}, f)

def search(query, k=5, per_doc=1):
    _ensure_loaded()
    qv = _l2_normalize(_model.encode([query]).astype("float32"))
    scores, idxs = _index.search(qv, min(50, 5*k))
    out, seen = [], set()
    for i, s in zip(idxs[0], scores[0]):
        m = _metas[i]
        key = m["doc_id"]
        if per_doc and key in seen:
            continue
        seen.add(key)
        out.append({
            "rank": len(out)+1,
            "score": float(s),
            "title": m["title"],
            "source": m["source"],
            "doc_id": key,
            "chunk_id": m.get("chunk_id", 0),
        })
        if len(out) == k:
            break
    return out

if __name__ == "__main__":
    build_index()
