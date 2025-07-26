from datetime import datetime

from brave_search import search
from github import get_git, get_name
from filesystem import save, load
from playwright_scrape import scrape
from sequential import think, compare
from notion import notion_save
from shared import get_query

#CHANGE personal.personal TO config
from config import BRAVE_API_KEY, GITHUB_TOKEN, SAVE_DIR, GPT_API, GPT_MODEL, NOTION_API_KEY, NOTION_PAD_ID

query = get_query()
limit = int(input("Limit: "))
github_links = search(query, BRAVE_API_KEY, limit)

repo_data = []

if github_links:
    for url in github_links:
        print(f"Fetching info for {url}... {github_links.index(url) + 1}/{len(github_links)}")
        data = get_git(url, GITHUB_TOKEN)
        if data:
            print(f"Fetching README for {url}... {github_links.index(url) + 1}/{len(github_links)}")
            readme, _ = scrape(url)
            if not readme:
                readme = "README not available"
        
            summary_html = think(GPT_API, data, readme, GPT_MODEL)
            data["summary"] = summary_html
            
            repo_data.append(data)
        else:
            print("Could not find info for {url}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fname = f"repos_{timestamp}.json"
    save(repo_data, fname, SAVE_DIR)
    print(f"Saved to {SAVE_DIR}repos_{timestamp}.json")
    
    print("Generating comparison...")
    comparison = compare(GPT_API, repo_data, GPT_MODEL)
    print(comparison)
    
    # for repo in repo_data:
    #     notion_save(NOTION_PAD_ID, NOTION_API_KEY, repo)
    
else:
    print("Nothing found.")