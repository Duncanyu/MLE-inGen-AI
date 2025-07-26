import requests

def search(query, api_key, limit = 5):
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": limit
    }
    
    response = requests.get(url, headers = headers, params = params)
    results = response.json()
    
    github_links = [
        item['url']
        for item in results.get('web', {}).get('results', [])
        if "github.com" in item['url']
    ]
    
    return github_links

