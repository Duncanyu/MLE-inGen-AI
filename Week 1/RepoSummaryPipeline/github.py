import requests

def get_git(url, token):
    link = url.split("/")
    repo = str(link[-1]); owner = str(link[-2])
    
    headers = {
        "Authorization": f"token {token}",
        "Accept" : "application/vnd.github.v3+json"
    }
    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers = headers).json()

    return {
        "name": response.get("name"),
        "description": response.get("description"),
        "stars": response.get("stargazers_count"),
        "language": response.get("language"),
        "url": response.get("html_url"),
        "last_updated": response.get("updated_at")
    }

def get_name(url):
    link = url.split("/")
    name = str(link[-1])