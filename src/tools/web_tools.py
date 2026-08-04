"""Web tools — keyless DuckDuckGo search and raw HTTP fetch.

Both tools are read-only GET requests; no API key is required. Network
work runs in a worker thread (``asyncio.to_thread``) so the event loop
never blocks.
"""

from __future__ import annotations

import asyncio
import logging
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)

# --- Baked-in defaults (no env vars; see Phase 8 plan) ---
_HTTP_TIMEOUT = 15.0
_DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_MAX_CHARS = 2000
# DDG serves results to browser-like UAs; python-requests' default UA is blocked.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _http_get(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: float = _HTTP_TIMEOUT,
) -> str:
    """Synchronous GET returning response text; raises ``ToolError`` on failure."""
    try:
        resp = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        )
    except requests.RequestException as exc:
        raise ToolError(f"Web request failed: {exc}") from exc
    if resp.status_code != 200:
        raise ToolError(f"HTTP {resp.status_code} from {url}")
    return resp.text


def _decode_ddg_url(href: str) -> str:
    """Unwrap a DuckDuckGo ``//duckduckgo.com/l/?uddg=...`` redirect URL."""
    if href.startswith("//"):
        href = "https:" + href
    parts = urlparse(href)
    if parts.netloc == "duckduckgo.com" and parts.path.startswith("/l/"):
        target = parse_qs(parts.query).get("uddg", [""])[0]
        if target:
            return target
    return href


class _DDGResultParser(HTMLParser):
    """Extract ``(title, url, snippet)`` triples from DDG html results.

    Keyed on the ``result__a`` / ``result__snippet`` CSS classes rather than
    element tags, so it survives markup tweaks (either an ``<a>`` or a
    ``<div>`` may carry the snippet class).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str, str]] = []
        self._title = ""
        self._url = ""
        self._snippet = ""
        self._mode: str | None = None  # "title" or "snippet"
        self._depth = 0
        self._buf: list[str] = []

    def _close_anchor(self) -> None:
        if self._mode == "title":
            self._title = " ".join("".join(self._buf).split())
        elif self._mode == "snippet":
            self._snippet = " ".join("".join(self._buf).split())
        self._mode = None
        self._buf = []

    def _flush(self) -> None:
        if self._title:
            self.results.append((self._title, self._url, self._snippet))
        self._title = self._url = self._snippet = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = {v for k, v in attrs if k == "class" and v}
        if "result__a" in classes or "result__snippet" in classes:
            if self._mode:
                self._close_anchor()
            if "result__a" in classes:
                self._flush()  # emit the previous completed result
                self._mode = "title"
                for k, v in attrs:
                    if k == "href" and v:
                        self._url = v
            else:
                self._mode = "snippet"
            self._depth = 1
            self._buf = []
        elif self._mode:
            self._depth += 1

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._mode:
            self._depth -= 1
            if self._depth == 0:
                self._close_anchor()

    def feed(self, data: str) -> None:
        super().feed(data)
        self._flush()


_SKIP_TAGS = {"head", "script", "style", "noscript", "iframe", "template"}


class _HTMLTextExtractor(HTMLParser):
    """Strip markup; drop script/style/head/nav content, keep readable text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(" ".join(self._chunks).split())


class WebSearchTool(Tool):
    """Search the web via DuckDuckGo's HTML endpoint (no API key)."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web and return a numbered list of titles, URLs, and short "
            "snippets. Uses DuckDuckGo; no API key required. Use for current or "
            "public information the agent does not already know."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": _DEFAULT_MAX_RESULTS,
                    "description": "Maximum results to return (1-10, default 5)",
                },
            },
            "required": ["query"],
        }

    async def run(  # type: ignore[override]
        self,
        query: str,
        max_results: int = _DEFAULT_MAX_RESULTS,
        **kwargs: Any,
    ) -> str:
        query = query.strip()
        if not query:
            raise ToolError("web_search: query must not be empty")
        max_results = max(1, min(10, max_results))

        html = await asyncio.to_thread(_http_get, _DDG_ENDPOINT, params={"q": query})

        lowered = html.lower()
        if "anomaly" in lowered or "challenge" in lowered:
            raise ToolError(
                "DuckDuckGo blocked the search (anti-bot challenge). Try again shortly."
            )

        parser = _DDGResultParser()
        parser.feed(html)
        results = parser.results[:max_results]
        if not results:
            return "No results found."

        lines: list[str] = []
        for i, (title, url, snippet) in enumerate(results, 1):
            url = _decode_ddg_url(url)
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
        return "\n".join(lines)


class WebFetchTool(Tool):
    """Fetch a URL and return its readable text (markup stripped)."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page over HTTP(S) and return its text with tags, scripts, "
            "and styles removed. Use to read article or documentation content. "
            "Only http/https URLs are allowed."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http:// or https:// URL to fetch",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 100,
                    "default": _DEFAULT_MAX_CHARS,
                    "description": "Max characters of extracted text to return",
                },
            },
            "required": ["url"],
        }

    async def run(  # type: ignore[override]
        self,
        url: str,
        max_chars: int = _DEFAULT_MAX_CHARS,
        **kwargs: Any,
    ) -> str:
        # Note on SSRF: we only gate the scheme. Blocking loopback/private IPs
        # would be a false sense of security here — the agent already has
        # read_file (and run_shell at safety >= 1), and requests follows
        # redirects / DNS rebinding defeats any pre-request host check unless
        # every resolved hop is validated, which requests doesn't expose.
        scheme = urlparse(url).scheme.lower()
        if scheme not in ("http", "https"):
            raise ToolError(f"web_fetch: only http/https URLs are allowed, got {url!r}")
        max_chars = max(100, min(10000, max_chars))

        html = await asyncio.to_thread(_http_get, url)
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()

        if len(text) > max_chars:
            return text[:max_chars] + f"\n[truncated to first {max_chars} chars]"
        return text
