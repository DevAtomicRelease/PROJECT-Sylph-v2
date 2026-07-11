import logging
import urllib.parse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

logger = logging.getLogger("sylph.tools.web")

async def search_web(query: str) -> list[dict]:
    """
    Search the web using DuckDuckGo and return the top 5 results.

    Args:
        query: The search terms.

    Returns:
        A list of dicts: {"title": str, "url": str, "snippet": str}
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        # DuckDuckGo HTML-only version is faster and easier to parse
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        logger.info("Searching web for query: '%s'", query)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, "html.parser")
        results = []
        for result in soup.select(".result")[:5]:
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            link_el = result.select_one(".result__url")

            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            
            # Find the actual anchor element to get the href
            a_el = result.select_one(".result__a")
            url_str = a_el["href"] if a_el and a_el.has_attr("href") else ""
            if not url_str and link_el:
                url_str = link_el.get_text(strip=True)

            if title or url_str:
                results.append({
                    "title": title,
                    "url": url_str,
                    "snippet": snippet
                })

        logger.info("Web search returned %d results", len(results))
        return results
    except Exception as e:
        logger.error("Web search failed: %s", e)
        raise e

async def read_page(url: str) -> str:
    """
    Fetch a web page and return its main text content.

    Args:
        url: The web page URL.

    Returns:
        The text content of the page.
    """
    try:
        logger.info("Reading web page: %s", url)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, "html.parser")

        # Strip scripting and styling
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()

        # Try to find main content elements
        main_content = None
        for selector in ["article", "main", "[role='main']", "#content", ".content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body if soup.body else soup

        # Extract text paragraphs
        paragraphs = main_content.find_all("p")
        text_lines = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
        
        if not text_lines:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = "\n\n".join(text_lines)

        # Truncate content to avoid token blowup
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated due to length]"

        logger.info("Successfully read %d characters from page", len(text))
        return text
    except Exception as e:
        logger.error("Failed to read page %s: %s", url, e)
        raise e
