from notion_client import Client

def notion_save(database_id, notion_token, repo):
    from notion_client import Client

    notion = Client(auth=notion_token)

    properties = {
        "Name": {
            "title": [
                {
                    "text": {
                        "content": repo.get("name", "Unnamed Repo")
                    }
                }
            ]
        },
        "Stars": {
            "number": repo.get("stars", 0)
        },
        "URL": {
            "url": repo.get("url", "")
        }
    }

    summary = repo.get("summary", "").strip()

    children = []
    if summary:
        for paragraph in summary.split("\n\n"):
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": paragraph}
                        }
                    ]
                }
            })

    notion.pages.create(
        parent={"database_id": database_id},
        properties=properties,
        children=children if children else None
    )
