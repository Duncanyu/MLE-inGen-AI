import sqlite3, json
from .paths import CHUNKS_JSONL, FTS_DB

def build_db():
    con = sqlite3.connect(str(FTS_DB))
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, doc_id TEXT, title TEXT, year TEXT, source TEXT, text TEXT)")
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, title, doc_id, content='chunks', content_rowid='id', tokenize='porter')")
    cur.execute("DELETE FROM chunks")
    cur.execute("DELETE FROM chunks_fts")
    rows = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append((rec.get("doc_id"), rec.get("title",""), rec.get("year",""), rec.get("source",""), rec["text"]))
    cur.executemany("INSERT INTO chunks(doc_id,title,year,source,text) VALUES (?,?,?,?,?)", rows)
    cur.execute("INSERT INTO chunks_fts(rowid, text, title, doc_id) SELECT id, text, title, doc_id FROM chunks")
    con.commit()
    con.close()

def bm25_search(query, k=10):
    con = sqlite3.connect(str(FTS_DB))
    cur = con.cursor()
    cur.execute("SELECT c.id, c.doc_id, c.title, c.source, bm25(chunks_fts) AS r FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ? ORDER BY r LIMIT ?", (query, k))
    out = []
    for row in cur.fetchall():
        out.append({"rowid": row[0], "doc_id": row[1], "title": row[2], "source": row[3], "bm25": float(row[4])})
    con.close()
    return out

if __name__ == "__main__":
    build_db()