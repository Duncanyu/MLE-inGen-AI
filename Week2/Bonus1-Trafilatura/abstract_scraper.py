import arxiv
import json
import requests
import trafilatura

search = arxiv.Search(
    query="cs.CL",
    max_results=200,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

client = arxiv.Client()

data = []
for result in client.results(search):
    abs_url = result.entry_id
    
    try:
        html = requests.get(abs_url, timeout=10).text
        cleaned = trafilatura.extract(html)
        
        abstract_text = cleaned if cleaned else result.summary
        
        data.append({
            "url": abs_url,
            "title": result.title.strip(),
            "abstract": abstract_text.strip(),
            "authors": [author.name for author in result.authors],
            "date": result.published.strftime("%Y-%m-%d")
        })
        
    except Exception as e:
        print(f"Failed to scrape {abs_url}: {e}")

with open("/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/arXiv/arxiv_clean.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"saved {len(data)} papers to arxiv_clean.json")
