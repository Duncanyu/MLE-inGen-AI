import json
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CHUNKS_PATH = Path("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/Homework/src/data/chunks.jsonl")
list_of_chunks = []
metas = []

with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        text = f"{rec.get('title', '')} — {rec['text']}"
        list_of_chunks.append(text)
        metas.append({
            "i": i,
            "doc_id": rec.get("doc_id"),
            "title": rec.get("title"),
            "source": rec.get("source"),
            "chunk_id": rec.get("chunk_id", 0),
        })

print(f"Loaded {len(list_of_chunks)} chunks")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(list_of_chunks, show_progress_bar=True).astype("float32")

def l2_normalize(x: np.ndarray):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

X = l2_normalize(embeddings)

dim = X.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(X)

INDEX_PATH = Path("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/Homework/src/data/index.faiss")
META_PKL   = Path("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/Homework/src/data/chunk_meta.pkl")

faiss.write_index(index, str(INDEX_PATH))
with META_PKL.open("wb") as f:
    pickle.dump({"metas": metas, "model_name": MODEL_NAME}, f)

print(f"Saved index to {INDEX_PATH}")
print(f"Saved metadata to {META_PKL}")

def search(query, k=5, per_doc=1):
    qv = l2_normalize(model.encode([query]).astype("float32"))
    scores, idxs = index.search(qv, min(50, 5*k))
    out, seen = [], set()
    for i, s in zip(idxs[0], scores[0]):
        m = metas[i]; key = m["doc_id"]
        if per_doc and key in seen: 
            continue
        seen.add(key); out.append({"rank": len(out)+1, "score": float(s), "title": m["title"], "source": m["source"]})
        if len(out) == k: break
    return out

if __name__ == "__main__":
    for q in ["instruction tuning", "supervised fine-tuning", "alignment with RLHF"]:
        print(f"\nQuery: {q}")
        for r in search(q, k=3):
            print(r)