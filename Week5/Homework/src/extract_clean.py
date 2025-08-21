import json, re, hashlib
from pathlib import Path
from tqdm import tqdm
import fitz
from .paths import DATA_DIR

META_PATH = DATA_DIR / "metadata.jsonl"
PDF_DIR   = DATA_DIR / "pdfs"
OUT_RAW   = DATA_DIR / "texts.jsonl"
OUT_DEDUP = DATA_DIR / "texts_dedup.jsonl"

def clean_page_text(t):
    if not t:
        return ""
    t = t.replace("\r", " ").replace("\t", " ")
    t = re.sub(r"[ \u00A0]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \u00A0]+", " ", t)
    return t.strip()

def strip_common_noise(full):
    keep = []
    for ln in full.splitlines():
        s = ln.strip()
        if not s:
            keep.append(s)
            continue
        if re.match(r"^arXiv:\d{4}\.\d{4,5}", s, flags=re.I):
            continue
        low = s.lower()
        if low.startswith("arxiv preprint") or low.startswith("preprint submitted to"):
            continue
        keep.append(s)
    return "\n".join(keep).strip()

def maybe_trim_references(t):
    m = re.search(r"\n\s*(REFERENCES|References|Bibliography)\s*\n", t)
    if not m:
        return t
    head, tail = t[:m.start()], t[m.start():]
    return head.strip() if len(tail) > 2000 else t

def normalize_for_hash(t):
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def hash_text(t):
    return hashlib.sha256(normalize_for_hash(t).encode("utf-8")).hexdigest()

def hash_title_year(title, date):
    year = date.split("-")[0] if date else ""
    key = f"{(title or '').strip().lower()}_{year}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        pages.append(clean_page_text(p.get_text("text")))
    page_count = len(pages)
    doc.close()
    full = "\n\n".join(pages)
    full = strip_common_noise(full)
    full = maybe_trim_references(full)
    return full, page_count

def resolve_pdf_path(rec):
    raw = rec.get("pdf_path", "")
    p = Path(raw) if raw else None
    if p and not p.is_absolute():
        p = (PDF_DIR / p).resolve()
    if not p or not p.exists():
        guess = PDF_DIR / f"{rec.get('id','')}.pdf"
        if guess.exists():
            p = guess
    return p

def main():
    assert META_PATH.exists(), f"missing: {META_PATH}"
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

    raw_count = 0
    with OUT_RAW.open("w", encoding="utf-8") as out_raw, META_PATH.open("r", encoding="utf-8") as meta_f:
        for line in tqdm(meta_f, desc="extracting"):
            rec = json.loads(line)
            pdf_path = resolve_pdf_path(rec)
            if not pdf_path or not pdf_path.exists():
                continue
            try:
                text, page_count = extract_pdf_text(pdf_path)
                if len(text) < 500:
                    continue
                out = {
                    "id": rec["id"],
                    "title": rec.get("title", ""),
                    "date": rec.get("date", ""),
                    "authors": rec.get("authors", []),
                    "source": rec.get("abs_url", ""),
                    "page_count": page_count,
                    "text": text,
                }
                out_raw.write(json.dumps(out, ensure_ascii=False) + "\n")
                raw_count += 1
            except Exception as e:
                print(f"failedl {rec.get('id','?')}: {e}")

    print(f"{raw_count} cleaned. {OUT_RAW}")

    seen_text = set()
    seen_title_year = set()
    kept = 0

    with OUT_RAW.open("r", encoding="utf-8") as f, OUT_DEDUP.open("w", encoding="utf-8") as g:
        for line in f:
            rec = json.loads(line)
            h1 = hash_text(rec["text"])
            h2 = hash_title_year(rec.get("title",""), rec.get("date",""))
            if h1 in seen_text or h2 in seen_title_year:
                continue
            seen_text.add(h1)
            seen_title_year.add(h2)
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept {kept} records in {OUT_DEDUP}")

if __name__ == "__main__":
    main()