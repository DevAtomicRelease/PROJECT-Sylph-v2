"""
Interactive web actions — Phase 5.

Unlike web_tool (fresh headless browser per scrape), this keeps ONE headed
browser + page alive so the agent can navigate, click, and type across steps
while the user watches. Gated by the 'web' domain. Non-destructive: it only
drives a browser; it never touches the local filesystem or system.

Best-effort locators (visible text / roles / placeholders) — arbitrary pages
vary, so each function returns a clear success or failure string.
"""

import logging

logger = logging.getLogger("sylph.tools.web_session")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
_ACTION_TIMEOUT = 8000  # ms


class WebSession:
    def __init__(self) -> None:
        self._pw = None
        self._browser = None
        self._page = None

    async def _ensure(self):
        if self._page is not None:
            return self._page
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=False)
        context = await self._browser.new_context(user_agent=_UA)
        self._page = await context.new_page()
        logger.info("WebSession browser launched (headed)")
        return self._page

    async def navigate(self, url: str) -> str:
        url = (url or "").strip()
        if not url:
            return "No URL given."
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            page = await self._ensure()
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            title = await page.title()
            return f"Opened '{title}' ({url})."
        except Exception as e:
            return f"Couldn't open {url}: {e}"

    async def click(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "What should I click?"
        try:
            page = await self._ensure()
            # Try button/link by accessible name, then any visible text.
            for locator in (
                page.get_by_role("button", name=text),
                page.get_by_role("link", name=text),
                page.get_by_text(text, exact=False),
            ):
                try:
                    await locator.first.click(timeout=_ACTION_TIMEOUT)
                    return f"Clicked '{text}'."
                except Exception:
                    continue
            return f"Couldn't find anything to click matching '{text}'."
        except Exception as e:
            return f"Click failed: {e}"

    async def type_text(self, text: str, into: str = "") -> str:
        try:
            page = await self._ensure()
            target = None
            into = (into or "").strip()
            if into:
                for locator in (page.get_by_label(into), page.get_by_placeholder(into)):
                    try:
                        await locator.first.wait_for(timeout=2000)
                        target = locator.first
                        break
                    except Exception:
                        continue
            if target is None:
                target = page.get_by_role("textbox").first
            await target.fill(text, timeout=_ACTION_TIMEOUT)
            return f"Typed into {'the ' + into if into else 'the field'}."
        except Exception as e:
            return f"Couldn't type: {e}"

    async def page_text(self) -> str:
        try:
            page = await self._ensure()
            text = await page.inner_text("body")
            return text[:6000] + ("\n[truncated]" if len(text) > 6000 else "")
        except Exception as e:
            return f"Couldn't read the page: {e}"

    async def close(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        finally:
            self._pw = self._browser = self._page = None


_session = WebSession()

# tool-facing wrappers
async def web_navigate(url: str) -> str:
    return await _session.navigate(url)

async def web_click(text: str) -> str:
    return await _session.click(text)

async def web_type(text: str, into: str = "") -> str:
    return await _session.type_text(text, into)

async def web_page_text() -> str:
    return await _session.page_text()

async def close_web_session() -> None:
    await _session.close()
