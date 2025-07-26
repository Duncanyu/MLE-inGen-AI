from playwright.sync_api import sync_playwright

def scrape(repo_url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(repo_url)

        try:
            readme_element = page.query_selector("article.markdown-body")
            if not readme_element:
                return None, None
            return readme_element.inner_text(), None
        finally:
            browser.close()