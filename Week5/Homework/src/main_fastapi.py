import os
import sys
import pickle
import subprocess
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .paths import DATA_DIR
from .faiss_script import search as vector_search
from .sqlite import bm25_search
from .hybrid import hybrid_search

ARXIV_QUERY = os.environ.get("ARXIV_QUERY", "cat:cs.CL")
ARXIV_MAX_RESULTS = int(os.environ.get("ARXIV_MAX_RESULTS", "200"))

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
STATIC_DIR = BASE / "static"

SCRIPTS = [
    "src.scrape",
    "src.extract_clean",
    "src.chunking",
    "src.faiss_script",
    "src.sqlite",
]

def run(script, env=None):
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run(
        [sys.executable, "-m", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed!!! {r.stderr}")
    return {"cmd": f"python -m {script}", "stdout": r.stdout, "stderr": r.stderr, "code": r.returncode}

def run_pipeline():
    env = os.environ.copy()
    env["ARXIV_QUERY"] = ARXIV_QUERY
    env["ARXIV_MAX_RESULTS"] = str(ARXIV_MAX_RESULTS)
    logs = []
    for s in SCRIPTS:
        logs.append(run(s, env=env))
    return {"status": "ok", "detail": "pipeline finished", "logs": logs}

app = FastAPI(title="Week5 Pipeline API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineRequest(BaseModel):
    query: str | None = None
    max_results: int | None = None

class PipelineResponse(BaseModel):
    status: str
    detail: str

class SearchResponseItem(BaseModel):
    rank: int
    score: float
    title: str | None = None
    source: str | None = None
    doc_id: str | None = None
    chunk_id: int | None = None

class KeywordResponseItem(BaseModel):
    doc_id: str | None = None
    title: str | None = None
    source: str | None = None
    bm25: float

class HybridResponseItem(BaseModel):
    rank: int
    doc_id: str
    title: str | None = None
    source: str | None = None
    score: float
    faiss: float | None = None
    bm25: float | None = None

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
    global ARXIV_QUERY, ARXIV_MAX_RESULTS
    if req.query:
        ARXIV_QUERY = req.query
    if req.max_results:
        ARXIV_MAX_RESULTS = int(req.max_results)
    try:
        result = run_pipeline()
        return PipelineResponse(status=result["status"], detail=result["detail"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

@app.get("/search")
def search_endpoint(q: str = Query(...), k: int = Query(5, ge=1, le=50), per_doc: int = Query(1, ge=0, le=1)):
    try:
        res = vector_search(q, k=k)
        out = []
        for i, r in enumerate(res):
            out.append({
                "rank": i + 1,
                "score": r.get("score", 0.0),
                "title": r.get("title"),
                "source": r.get("source"),
                "doc_id": r.get("doc_id"),
                "chunk_id": r.get("chunk_id"),
            })
        return [SearchResponseItem(**x) for x in out]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/keyword_search")
def keyword_search_endpoint(q: str = Query(...), k: int = Query(5, ge=1, le=50)):
    try:
        res = bm25_search(q, k=k)
        out = []
        for r in res:
            out.append({
                "doc_id": r.get("doc_id"),
                "title": r.get("title"),
                "source": r.get("source"),
                "bm25": r.get("bm25"),
            })
        return [KeywordResponseItem(**x) for x in out]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/hybrid_search")
def hybrid_search_endpoint(q: str = Query(...), k: int = Query(5, ge=1, le=50), alpha: float = Query(0.6, ge=0.0, le=1.0), per_doc: int = Query(1, ge=0, le=1)):
    try:
        res = hybrid_search(q, k=k, per_doc=per_doc, alpha=alpha)
        out = []
        for i, r in enumerate(res):
            out.append({
                "rank": i + 1,
                "doc_id": r.get("doc_id"),
                "title": r.get("title"),
                "source": r.get("source"),
                "score": r.get("score", 0.0),
                "faiss": r.get("faiss"),
                "bm25": r.get("bm25"),
            })
        return [HybridResponseItem(**x) for x in out]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
