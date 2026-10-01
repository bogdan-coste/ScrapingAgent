from playwright.async_api import Browser, async_playwright

_playwright = None
_browser: Browser | None = None


async def get_browser() -> Browser:
    """
        Return the one browser shared by the whole process, started on first use.
    """
    global _playwright, _browser

    if _browser is None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch()

    return _browser
