import os
import sys
import pickle
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    faiss = None
    SentenceTransformer = None
    np = None

ARXIV_QUERY = os.environ.get("ARXIV_QUERY", "cat:cs.CL")
ARXIV_MAX_RESULTS = int(os.environ.get("ARXIV_MAX_RESULTS", "200"))
SCRIPTS = ["scrape.py", "extract_clean.py", "chunking.py", "faiss_script.py"]

BASE = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [
    Path("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/Homework/src/data"),
    BASE / "data",
]

def first_existing(*paths: Path):
    for p in paths:
        if p and p.exists():
            return p
    return None

def data_path(rel: str):
    return first_existing(*[(d / rel) for d in DATA_DIR_CANDIDATES])

def run(script: str, env=None):
    print(f"\n == Running {script}... ==")
    subprocess.run([sys.executable, str(BASE / script)], check=True, env=env)

def run_pipeline():
    env = os.environ.copy()
    env["ARXIV_QUERY"] = ARXIV_QUERY
    env["ARXIV_MAX_RESULTS"] = str(ARXIV_MAX_RESULTS)
    for script in SCRIPTS:
        run(script, env=env)
    return {"status": "ok", "detail": "pipeline finished"}

_index = None
_model = None
_metas = None
_model_name = None

def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def ensure_search_loaded():
    global _index, _model, _metas, _model_name
    if _index is not None:
        return
    if any(x is None for x in (faiss, SentenceTransformer, np)):
        raise RuntimeError("Search dependencies not installed. Install: faiss-cpu, sentence-transformers, numpy")
    index_path = data_path("index.faiss")
    meta_pkl = data_path("chunk_meta.pkl")
    if not index_path or not index_path.exists() or not meta_pkl or not meta_pkl.exists():
        raise FileNotFoundError("index.faiss or chunk_meta.pkl not found. Run the pipeline first.")
    _index = faiss.read_index(str(index_path))
    with open(meta_pkl, "rb") as f:
        blob = pickle.load(f)
    _metas = blob.get("metas", [])
    _model_name = blob.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    _model = SentenceTransformer(_model_name)

def search(query: str, k: int = 5, per_doc: int = 1):
    ensure_search_loaded()
    qv = _model.encode([query]).astype("float32")
    qv = l2_normalize(qv)
    scores, idxs = _index.search(qv, min(50, 5 * k))
    out, seen = [], set()
    for i, s in zip(idxs[0], scores[0]):
        m = _metas[i]
        key = m.get("doc_id")
        if per_doc and key in seen:
            continue
        seen.add(key)
        out.append({
            "rank": len(out) + 1,
            "score": float(s),
            "title": m.get("title"),
            "source": m.get("source"),
            "doc_id": key,
            "chunk_id": m.get("chunk_id", 0),
        })
        if len(out) == k:
            break
    return out

app = FastAPI(title="Week4 Pipeline API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineRequest(BaseModel):
    query: Optional[str] = None
    max_results: Optional[int] = None

class PipelineResponse(BaseModel):
    status: str
    detail: str

class SearchResponseItem(BaseModel):
    rank: int
    score: float
    title: Optional[str]
    source: Optional[str]
    doc_id: Optional[str]
    chunk_id: int

STATIC_DIR = BASE / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail=f"Missing {index}")
    return FileResponse(index)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/pipeline", response_model=PipelineResponse)
def pipeline(req: PipelineRequest):
    global ARXIV_QUERY, ARXIV_MAX_RESULTS, _index, _model, _metas
    if req.query:
        ARXIV_QUERY = req.query
    if req.max_results:
        ARXIV_MAX_RESULTS = int(req.max_results)
    _index = _model = _metas = None
    try:
        result = run_pipeline()
        return PipelineResponse(**result)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

@app.get("/search", response_model=List[SearchResponseItem])
def search_endpoint(
    q: str = Query(..., description="search query"),
    k: int = Query(5, ge=1, le=50),
    per_doc: int = Query(1, ge=0, le=1, description="1=unique per doc, 0=allow multiple chunks"),
):
    try:
        results = search(q, k=k, per_doc=per_doc)
        return [SearchResponseItem(**r) for r in results]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)