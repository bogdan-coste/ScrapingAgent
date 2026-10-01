from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.async_api import (
    Browser,
    Playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from playwright_stealth import Stealth


class PageLoadError(RuntimeError):
    """
        The target page could not be loaded.
    """

class WebpageInspector:

    def __init__(
        self,
        url: str,
        headless: bool = True,
        timeout_ms: int = 30_000,
        user_agent: str | None = None,
        browser: Browser | None = None
    ):

        # set the url page of
        # the webpage we want to explore with playwright
        self.url = url

        self.headless = headless
        self.timeout_ms = timeout_ms
        self.user_agent = user_agent

        # dynamically modifies the variables
        # so an automated activity looks normal
        self.stealth = Stealth()

        # browser esntry is to inject an already running browser
        # to avoid opening a separate browser for each website
        # that needs to be inspected
        self._browser = browser

        # a check to know if the current instance
        # already owns a browser
        self._owns_browser = browser is None

        # the handle to the engine that launches
        # and drives browsers
        self._playwright: Playwright | None = None

        # every url the agent has visited, mapped to that page's contents
        self.webpage_history: dict[str, str] = {}

    async def load_contents(self) -> str:
        """
            Load the inspector's own URL and return its preprocessed contents.
        """
        return await self.go_to(self.url)

    async def go_to(self, url: str) -> str:
        """
            Point the browser at ``url`` (e.g. a link the LLM picked from the
            HTML) and return the preprocessed contents of that page.
        """
        if self._browser is None:
            raise RuntimeError(
                "WebpageInspector is not started; use `async with` or call start() first."
            )

        # get the entire, rendered HTML page
        context = await self._browser.new_context(user_agent=self.user_agent)

        try:
            await self.stealth.apply_stealth_async(context)

            page = await context.new_page()

            try:
                await page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
            except PlaywrightTimeoutError as exc:
                raise PageLoadError(f"Timed out after {self.timeout_ms}ms loading {url}") from exc
            except PlaywrightError as exc:
                # Links the LLM picks may point at non-HTML resources (PDF, DOCX,
                # XLSX…). Chromium begins a download instead of navigating, which
                # Playwright reports as "Download is starting". That is not a
                # failure: the link is a document, not a browsable page.
                if "Download is starting" in str(exc):
                    return (
                        f"'{url}' is a downloadable file (e.g. PDF/DOCX), not an HTML page, "
                        "so it cannot be navigated. Treat it as a document link instead."
                    )

                raise PageLoadError(f"Failed to load {url}: {exc}") from exc

            html_page = await page.content()
            contents = self.preprocess(html_page)

            # the browser is now on this page: remember it and its contents
            self.url = url
            self.webpage_history[url] = contents

            return contents

        finally:
            await context.close()

    def preprocess(self, html_page: str) -> str:
        """
            Strip everything that costs tokens but carries no content:
            <script>/<noscript> blocks, <style> blocks, stylesheet <link>s and
            inline style="" attributes.
        """

        soup = BeautifulSoup(html_page, "lxml")

        for tag in soup.find_all(["script", "noscript", "style"]):
            tag.decompose()

        for link in soup.find_all("link", rel="stylesheet"):
            link.decompose()

        for tag in soup.find_all(style=True):
            del tag["style"]

        return str(soup)

    def visited_urls(self) -> list[str]:
        """
            The URLs the agent has visited so far, in visit order.
        """
        return list(self.webpage_history)


def extract_links(html: str, base_url: str = "") -> list[str]:
    """
        Return the unique links found in an HTML document.

        Relative hrefs are resolved against ``base_url`` (usually the page URL),
        so the result can be followed directly. Purely in-page anchors and
        non-navigational schemes (javascript:, mailto:, tel:) are skipped.
    """
    soup = BeautifulSoup(html, "lxml")

    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        links.append(urljoin(base_url, href) if base_url else href)

    # de-duplicate, preserving order
    return list(dict.fromkeys(links))
