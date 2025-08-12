import arxiv, os, json, time
from pathlib import Path
from tqdm import tqdm
import os

QUERY = os.environ.get("ARXIV_QUERY", "cat:cs.CL")
MAX_RESULTS = int(os.environ.get("ARXIV_MAX_RESULTS", "200"))

OUT_DIR = Path("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/Homework/src/data/pdfs")
META_PATH = Path("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week4/Homework/src/data/metadata.jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RESULTS = 50

search = arxiv.Search(
    query=QUERY,
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

client = arxiv.Client(page_size=50, delay_seconds=0.5, num_retries=5)

downloaded = 0
with META_PATH.open("w", encoding="utf-8") as meta_f:
    for res in tqdm(client.results(search), total=MAX_RESULTS):
        arxiv_id = res.get_short_id()
        pdf_path = OUT_DIR / f"{arxiv_id}.pdf"

        meta_f.write(json.dumps({
            "id": arxiv_id,
            "title": (res.title or "").strip(),
            "authors": [a.name for a in res.authors],
            "date": res.published.strftime("%Y-%m-%d"),
            "primary_category": res.primary_category,
            "pdf_path": str(pdf_path),
            "abs_url": res.entry_id
        }, ensure_ascii=False) + "\n")

        if pdf_path.exists():
            continue

        try:
            res.download_pdf(dirpath=str(OUT_DIR), filename=f"{arxiv_id}.pdf")
            downloaded += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"Skipped {arxiv_id}: {e}")

print(f"Downloaded: {downloaded}")
