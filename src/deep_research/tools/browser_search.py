"""Browser search tool: headless Chrome scraping, no search API required.

Uses Playwright to drive the locally installed Chrome (``channel="chrome"``) so
there is no browser download and no API key. Pipeline per query:

1. Scrape the DuckDuckGo HTML endpoint for organic results (title, URL, snippet).
2. Visit the top pages and pull readable text from ``<body>`` (scripts/styles
   stripped, images/fonts/media blocked to keep pages fast).

This is deliberately the "hard way" compared to Exa — it is the honest
representation of what a search API does for you, and it makes the tool
swap-in fact visible in code. Trades-off: Playwright sync objects are not
thread-safe across the research fan-out, so scraping is serialized behind a
lock. Fine for a local single-worker demo; use Exa for real throughput.
"""

from __future__ import annotations

import concurrent.futures
from urllib.parse import quote_plus, unquote

from ..errors import SearchError
from ..logging_config import get_logger
from ..models import SearchHit
from .base import SearchTool

logger = get_logger(__name__)

DDG_URL = "https://html.duckduckgo.com/html/?q={query}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_BLOCKED_RESOURCE_TYPES = {"image", "font", "media"}


class BrowserSearchTool(SearchTool):
    name = "browser"

    def __init__(
        self,
        num_results: int = 5,
        timeout: float = 15.0,
        max_page_chars: int = 6000,
        max_pages_to_read: int = 4,
        channel: str = "chrome",
    ) -> None:
        self.num_results = num_results
        self.timeout_s = timeout
        self.max_page_chars = max_page_chars
        self.max_pages_to_read = max_pages_to_read
        self.channel = channel
        # Playwright's sync API is thread-affine: objects must be created and
        # used on the same thread. The research fan-out calls search() from a
        # ThreadPoolExecutor, so all scraping is funneled through one dedicated
        # thread. Fine for a local single-worker demo; use Exa for throughput.
        self._scrape_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="browser-search"
        )
        self._playwright = None
        self._browser = None
        self._context = None

    # -- lifecycle (runs on the scrape thread) ----------------------------
    def _ensure_browser(self):
        """Must be called on the scrape thread; lazily launches Chrome there."""
        if self._context is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as exc:  # pragma: no cover - environment guard
                raise SearchError(
                    "playwright is required for the browser provider: pip install playwright"
                ) from exc

            self._playwright = sync_playwright().start()
            try:
                # Installed Chrome first; fall back to the bundled chromium.
                self._browser = self._playwright.chromium.launch(
                    channel=self.channel, headless=True
                )
            except Exception:
                logger.warning("Chrome channel unavailable; falling back to bundled chromium")
                self._browser = self._playwright.chromium.launch(headless=True)
            self._context = self._browser.new_context(user_agent=USER_AGENT)
            self._context.set_default_timeout(self.timeout_s * 1000)
        return self._context

    def _search_on_scrape_thread(self, query: str, limit: int) -> list[SearchHit]:
        context = self._ensure_browser()
        page = context.new_page()
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in _BLOCKED_RESOURCE_TYPES
            else route.continue_(),
        )
        try:
            hits = self._scrape_results(page, query, limit)
            for hit in hits[: self.max_pages_to_read]:
                hit.text = self._readable_text(page, hit.url)
            return hits
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(f"browser search failed for {query!r}: {exc}") from exc
        finally:
            page.close()

    def _shutdown(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def close(self) -> None:
        """Shut Chrome down on its own thread, but never hang forever on it."""
        try:
            self._scrape_pool.submit(self._shutdown).result(timeout=10)
        except Exception:  # timeout or dead driver: abandon the thread
            pass
        self._scrape_pool.shutdown(wait=False)

    # -- search -----------------------------------------------------------
    def search(self, query: str, num_results: int = 5) -> list[SearchHit]:
        limit = min(num_results or self.num_results, self.num_results or num_results)
        future = self._scrape_pool.submit(self._search_on_scrape_thread, query, limit)
        try:
            return future.result()
        except SearchError as first:
            # Long-lived Chrome sessions die eventually (crashed tab, driver
            # transport dropped). Playwright's close() on a dead connection can
            # hang forever, so we do NOT try to shut it down gracefully: the
            # hung thread is abandoned and a fresh pool + browser is built.
            if "connection closed" not in str(first).lower():
                raise
            logger.warning("browser driver died; rebuilding browser and retrying %r", query)
            self._rebuild_after_crash()
            retry = self._scrape_pool.submit(self._search_on_scrape_thread, query, limit)
            try:
                return retry.result()
            except SearchError:
                raise
        except concurrent.futures.CancelledError:  # pragma: no cover
            raise SearchError("browser search cancelled") from None

    def _rebuild_after_crash(self) -> None:
        """Drop the dead browser without waiting on it; build a fresh pool."""
        self._playwright = self._browser = self._context = None  # leaked, never touched again
        self._scrape_pool.shutdown(wait=False)
        self._scrape_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="browser-search"
        )

    def _scrape_results(self, page, query: str, limit: int) -> list[SearchHit]:
        page.goto(DDG_URL.format(query=quote_plus(query)), wait_until="domcontentloaded")
        try:
            page.wait_for_selector("div.result__body", timeout=5_000)
        except Exception:
            pass  # parse whatever rendered

        hits: list[SearchHit] = []
        for body in page.locator("div.result__body").all()[: limit * 2]:
            link = body.locator("a.result__a").first
            url = link.get_attribute("href") or ""
            # DDG wraps outbound links: //duckduckgo.com/l/?uddg=<encoded>&...
            if "uddg=" in url:
                url = unquote(url.split("uddg=")[1].split("&")[0])
            if not url.startswith("http"):
                continue
            try:
                title = link.inner_text().strip()
            except Exception:
                title = ""
            snippet = ""
            sn = body.locator("a.result__snippet")
            if sn.count():
                try:
                    snippet = sn.first.inner_text().strip()
                except Exception:
                    snippet = ""
            hits.append(
                SearchHit(url=url, title=title, snippet=snippet, score=1.0 - len(hits) * 0.05)
            )
            if len(hits) >= limit:
                break
        logger.info("browser search %r -> %d results", query, len(hits))
        return hits

    def _readable_text(self, page, url: str) -> str:
        """Fetch one page and return its visible text, '' on any failure."""
        try:
            page.goto(url, wait_until="domcontentloaded")
            return page.evaluate(
                """() => {
                    for (const tag of ['script', 'style', 'noscript', 'svg']) {
                        document.querySelectorAll(tag).forEach((n) => n.remove());
                    }
                    const candidates = [
                        document.querySelector('main'),
                        document.querySelector('article'),
                        document.querySelector('[role=main]'),
                        document.querySelector('#content'),
                    ].filter(Boolean);
                    let best = document.body;
                    let bestLen = 0;
                    for (const node of candidates.concat([document.body])) {
                        const len = (node.innerText || '').length;
                        if (len > bestLen) { best = node; bestLen = len; }
                    }
                    return best.innerText;
                }"""
            )[: self.max_page_chars]
        except Exception as exc:
            logger.warning("browser could not read %s: %s", url, exc)
            return ""
