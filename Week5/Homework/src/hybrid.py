import pickle, sqlite3, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from .paths import FAISS_INDEX, FAISS_METAS, FTS_DB

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_index = None
_metas = None
_model = None

def _load_faiss():
    global _index, _metas, _model
    if _index is None:
        _index = faiss.read_index(str(FAISS_INDEX))
        with open(FAISS_METAS, "rb") as f:
            _metas = pickle.load(f)["metas"]
        _model = SentenceTransformer(MODEL_NAME)

def _faiss_search(query, k=20, per_doc=1):
    _load_faiss()
    qv = _model.encode([query]).astype("float32")
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    scores, idxs = _index.search(qv, min(50, 5*k))
    out, seen = [], set()
    for i, s in zip(idxs[0], scores[0]):
        m = _metas[i]
        did = m["doc_id"]
        if per_doc and did in seen:
            continue
        seen.add(did)
        out.append({"doc_id": did, "title": m["title"], "source": m["source"], "faiss": float(s)})
        if len(out) == k:
            break
    return out

def _fts_search(query, k=20, per_doc=1):
    con = sqlite3.connect(str(FTS_DB))
    cur = con.cursor()
    cur.execute(
        "SELECT c.id, c.doc_id, c.title, c.source, bm25(chunks_fts) AS r "
        "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY r LIMIT ?",
        (query, k*5)
    )
    out, seen = [], set()
    for row in cur.fetchall():
        did = row[1]
        if per_doc and did in seen:
            continue
        seen.add(did)
        out.append({"rowid": row[0], "doc_id": did, "title": row[2], "source": row[3], "bm25": float(row[4])})
        if len(out) == k:
            break
    con.close()
    return out

def _minmax(xs):
    xs = list(xs)
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]

def hybrid_search(query, k=5, per_doc=1, alpha=0.6):
    v = _faiss_search(query, k=max(20, 4*k), per_doc=per_doc)
    t = _fts_search(query, k=max(20, 4*k), per_doc=per_doc)
    by_id = {}
    for r in v:
        by_id.setdefault(r["doc_id"], {}).update(r)
    for r in t:
        by_id.setdefault(r["doc_id"], {}).update(r)
    fa = [d.get("faiss", 0.0) for d in by_id.values()]
    bm = [d.get("bm25", 1e6) for d in by_id.values()]
    fa_n = _minmax(fa)
    bm_n = _minmax([-x for x in bm])
    merged = []
    for (doc_id, d), fz, bz in zip(by_id.items(), fa_n, bm_n):
        merged.append({
            "doc_id": doc_id,
            "title": d.get("title",""),
            "source": d.get("source",""),
            "faiss": d.get("faiss"),
            "bm25": d.get("bm25"),
            "score": float(alpha*fz + (1-alpha)*bz)
        })
    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged[:k]