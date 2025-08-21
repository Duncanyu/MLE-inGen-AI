import json
from pathlib import Path
from tqdm import tqdm
from .paths import DATA_DIR, CHUNKS_JSONL

IN_PATH = DATA_DIR / "texts_dedup.jsonl"
OUT_PATH = CHUNKS_JSONL
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def chunk_text(text, max_tokens=512, overlap=50):
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if i + max_tokens >= len(tokens):
            break
    return chunks

list_of_chunks = []
chunks_by_doc = {}
n_docs = 0
n_chunks = 0

with OUT_PATH.open("w", encoding="utf-8") as out_f, IN_PATH.open("r", encoding="utf-8") as in_f:
    for line in tqdm(in_f, desc="chunking"):
        rec = json.loads(line)
        doc_chunks = chunk_text(rec["text"], max_tokens=512, overlap=50)
        if not doc_chunks:
            continue
        for idx, ch in enumerate(doc_chunks):
            out_f.write(json.dumps({
                "doc_id": rec["id"],
                "title": rec.get("title"),
                "date": rec.get("date"),
                "source": rec.get("source"),
                "chunk_id": idx,
                "text": ch
            }, ensure_ascii=False) + "\n")
        chunks_by_doc[rec["id"]] = doc_chunks
        list_of_chunks.extend(doc_chunks)
        n_docs += 1
        n_chunks += len(doc_chunks)

print(f"Documents processed: {n_docs}")
print(f"Chunks written: {n_chunks}")