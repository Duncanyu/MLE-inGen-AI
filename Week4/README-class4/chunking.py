import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredFileLoader, Docx2txtLoader
)

INPUT_DIR = Path("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/README-class4/data/raw")
OUTPUT_PATH = Path("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/README-class4/data/interim/chunks.jsonl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SEPARATORS = [
    "\n\n", "\n", ". ", "? ", "! ",
    "; ", ": ", ", ", " ", ""
]

@dataclass
class Chunk:
    content: str
    source: str
    page: int | None
    order: int

def load_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext in [".txt", ".md"]:
        return TextLoader(str(path), autodetect_encoding=True).load()
    if ext in [".docx"]:
        return Docx2txtLoader(str(path)).load()
    return UnstructuredFileLoader(str(path)).load()

def normalize(text: str):
    t = text.replace("‐\n", "")
    t = t.replace("-\n", "")
    t = t.replace("\u00ad", "")
    return t

def main():
    docs = []
    for p in sorted(INPUT_DIR.glob("**/*")):
        if not p.is_file():
            continue
        try:
            docs.extend(load_file(p))
            print(f"Loaded: {p}")
        except Exception as e:
            print(f"skipped {p} ({e})")
    for d in docs:
        d.page_content = normalize(d.page_content or "")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        add_start_index=True,
    )
    split_docs = splitter.split_documents(docs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, d in enumerate(split_docs):
            meta = d.metadata or {}
            chunk = Chunk(
                content=d.page_content.strip(),
                source=str(meta.get("source") or meta.get("file_path") or "unknown"),
                page=meta.get("page") if isinstance(meta.get("page"), int) else None,
                order=i,
            )
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(f"writtne {len(split_docs)} chunks to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
