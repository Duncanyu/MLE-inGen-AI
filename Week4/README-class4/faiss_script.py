import os
import json
from pathlib import Path
from typing import List

import faiss
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from keys import OPENAI_KEY
except Exception:
    OPENAI_KEY = None

BASE = Path(__file__).resolve().parent
CHUNKS = BASE / "data" / "interim" / "chunks.jsonl"

EMBED_MODEL_OPENAI = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

def load_chunks(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing chunks file: {path}")
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            docs.append(
                Document(
                    page_content=c["content"],
                    metadata={
                        "source": c.get("source"),
                        "page": c.get("page"),
                        "order": c.get("order"),
                    },
                )
            )
    if not docs:
        raise RuntimeError("rerun chunking")
    return docs

def get_embedder():
    if OPENAI_KEY and OPENAI_KEY.strip():
        print("openai: ", EMBED_MODEL_OPENAI)
        return OpenAIEmbeddings(model=EMBED_MODEL_OPENAI, api_key=OPENAI_KEY)
    print("using HF embeddings: sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_store():
    docs = load_chunks(CHUNKS)
    embedder = get_embedder()
    return FAISS.from_documents(docs, embedder)

def main():
    store = build_store()

    if OPENAI_KEY and OPENAI_KEY.strip():
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_KEY)
    else:
        raise RuntimeError(
            "no OPENAI_KEY set in keys.py. Add your key or switch LLM to a local option."
        )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    while True:
        q = input("\nask a question (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break
        resp = qa({"query": q})
        print("\nanswer:\n", resp["result"])
        print("\nsources:")
        for d in resp["source_documents"]:
            src = d.metadata.get("source")
            page = d.metadata.get("page")
            print(f" • {src}" + (f" (p.{page})" if page is not None else ""))

if __name__ == "__main__":
    main()