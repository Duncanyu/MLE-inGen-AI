import json
import os
import re
from pathlib import Path
from langdetect import detect, DetectorFactory
from datasketch import MinHash, MinHashLSH

DetectorFactory.seed = 0

arxiv_path = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Bonus1-Trafilatura/arxiv_clean.json"
pdf_ocr_folder = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Bonus2-PDFs/pdf_ocr"
transcripts_path = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Bonus3-ASR/talks_transcripts.jsonl"

output_corpus_path = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Bonus4-Cleanup/clean_corpus.txt"
stats_path = "stats.md"

docs = []

with open(arxiv_path, "r", encoding="utf-8") as f:
    for item in json.load(f):
        docs.append(item["abstract"])

for txt_file in Path(pdf_ocr_folder).glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
        docs.append(f.read())

with open(transcripts_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        docs.append(obj["text"])

total_docs_before = len(docs)
print(f"loaded {total_docs_before} documents.")

lang_filtered = []
for d in docs:
    try:
        if detect(d) == "en":
            lang_filtered.append(d)
    except:
        continue

print(f"filter: {len(lang_filtered)} docs kept.")

def clean_html(text):
    return re.sub(r"<.*?>", "", text)

cleaned_docs = [clean_html(d) for d in lang_filtered]

def minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in set(text.split()):
        m.update(word.encode("utf8"))
    return m

lsh = MinHashLSH(threshold=0.7, num_perm=128)
unique_docs = []
for i, doc in enumerate(cleaned_docs):
    mh = minhash(doc)
    if not lsh.query(mh):
        lsh.insert(f"doc{i}", mh)
        unique_docs.append(doc)

print(f"deduplication: {len(unique_docs)} unique docs kept.")

email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
phone_re = re.compile(r"\+?\d[\d\s().-]{7,}\d")
cc_re = re.compile(r"(?:\d[ -]*?){13,16}")

def remove_pii(text):
    text = email_re.sub("[EMAIL]", text)
    text = phone_re.sub("[PHONE]", text)
    text = cc_re.sub("[CREDIT_CARD]", text)
    return text

pii_cleaned = [remove_pii(d) for d in unique_docs]

def remove_repetitive(text, n=3):
    words = text.split()
    filtered = []
    for i, w in enumerate(words):
        if i >= n and all(words[i-j] == w for j in range(1, n)):
            continue
        filtered.append(w)
    return " ".join(filtered)

final_docs = [remove_repetitive(d) for d in pii_cleaned]

with open(output_corpus_path, "w", encoding="utf-8") as f:
    for d in final_docs:
        f.write(d.strip() + "\n")

tokens_before = sum(len(d.split()) for d in docs)
tokens_after = sum(len(d.split()) for d in final_docs)
removal_pct = round((1 - tokens_after / tokens_before) * 100, 2)

with open(stats_path, "w", encoding="utf-8") as f:
    f.write(f"# Cleaning Stats\n")
    f.write(f"- Total documents before: {total_docs_before}\n")
    f.write(f"- Total documents after: {len(final_docs)}\n")
    f.write(f"- Tokens before: {tokens_before}\n")
    f.write(f"- Tokens after: {tokens_after}\n")
    f.write(f"- Removal percentage: {removal_pct}%\n")

print(f"Clleaning complete. Saved {len(final_docs)} docs to {output_corpus_path}")
