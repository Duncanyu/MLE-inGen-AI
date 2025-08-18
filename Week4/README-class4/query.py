import pickle
from pathlib import Path
import numpy as np, faiss

BASE = Path(__file__).resolve().parent
INDEX = faiss.read_index(str(BASE / "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/README-class4/data/index/faiss.index"))
meta = pickle.load(open(BASE / "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/README-class4/data/index/meta.pkl", "rb"))
texts = meta["texts"]

def search(q: str, k: 5):
    from faiss_script import get_embedder
    embed_batch, dim, _ = get_embedder()
    v = embed_batch([q]).astype("float32")
    faiss.normalize_L2(v)
    D, I = INDEX.search(v, k)
    return [(float(D[0][j]), int(I[0][j]), texts[int(I[0][j])]) for j in range(len(I[0]))]

if __name__ == "__main__":
    for score, idx, text in search("what python/ai projects have I done?", k=5):
        print(f"\n== score {score:.3f} | idx {idx} ==\n{text[:400]}...")
