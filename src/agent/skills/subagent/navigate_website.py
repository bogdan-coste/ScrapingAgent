from src.ingestion.browser import get_browser
from src.ingestion.page_contents import WebpageInspector, extract_links

# Overwritten with this subagent's url when the skill is loaded
# (SkillManager.shared_resources).
url: str = ""

_page: WebpageInspector | None = None


async def _inspector() -> WebpageInspector:
    """
        This subagent's inspector, built lazily with the shared browser.
    """
    global _page

    if _page is None:
        _page = WebpageInspector(url=url, browser=await get_browser())

    return _page


async def open_page() -> str:
    """
    Load this subagent's page and return its cleaned HTML.
    """
    inspector = await _inspector()

    return await inspector.load_contents()


async def get_links() -> list[str]:
    """
    Return the links on the current page, as absolute URLs.
    """
    inspector = await _inspector()

    return extract_links(await inspector.load_contents(), base_url=inspector.url)


async def follow(link: str) -> str:
    """
    Follow a link and return the new page's cleaned HTML.
    """
    inspector = await _inspector()

    return await inspector.go_to(link)


def visited_urls() -> list[str]:
    """
    List the URLs visited so far, in visit order.
    """
    if _page is None:
        return []

    return _page.visited_urls()
