"""Tests for the web search and fetch tools (network-free)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import requests

from src.tools import web_tools
from src.tools.base import ToolError
from src.tools.registry import ToolRegistry
from src.tools.web_tools import WebFetchTool, WebSearchTool


def _ddg_html(titles: list[str]) -> str:
    """Render DDG-style result blocks for *titles* (one result each)."""
    blocks = []
    for t in titles:
        blocks.append(
            f'<div class="result results_links web-result">'
            f'<div class="result__title">'
            f'<a rel="nofollow" class="result__a" '
            f'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F{t}">{t}</a>'
            f"</div>"
            f'<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">'
            f"snippet about {t}</a>"
            f"</div>"
        )
    return "<html><body>" + "".join(blocks) + "</body></html>"


def _numbered_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if re.match(r"^\d+\. ", line)]


class TestWebSearchToolContract:
    def test_name(self) -> None:
        assert WebSearchTool().name == "web_search"

    def test_description_mentions_search(self) -> None:
        description = WebSearchTool().description
        assert "search" in description.lower()
        assert "web" in description.lower()

    def test_schema_requires_query(self) -> None:
        schema = WebSearchTool().input_schema
        assert schema["type"] == "object"
        assert schema["required"] == ["query"]
        assert "query" in schema["properties"]

    def test_schema_max_results_bounds(self) -> None:
        max_results = WebSearchTool().input_schema["properties"]["max_results"]
        assert max_results["minimum"] == 1
        assert max_results["maximum"] == 10
        assert max_results["default"] == 5


class TestWebSearchToolRun:
    async def test_happy_path_unwraps_ddg_redirect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: _ddg_html(["Alpha", "Beta"]))
        result = await WebSearchTool().run(query="alpha")
        assert "1. Alpha" in result
        assert "2. Beta" in result
        assert "https://example.com/Alpha" in result
        assert "snippet about Alpha" in result
        assert "snippet about Beta" in result

    async def test_max_results_caps_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: _ddg_html(["A", "B", "C"]))
        result = await WebSearchTool().run(query="q", max_results=2)
        assert len(_numbered_lines(result)) == 2
        assert "3. " not in result

    async def test_max_results_zero_clamps_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: _ddg_html(["A", "B", "C"]))
        result = await WebSearchTool().run(query="q", max_results=0)
        assert len(_numbered_lines(result)) == 1
        assert "1. A" in result

    async def test_max_results_upper_clamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        titles = [f"R{i}" for i in range(12)]
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: _ddg_html(titles))
        result = await WebSearchTool().run(query="q", max_results=99)
        assert len(_numbered_lines(result)) == 10

    async def test_no_results_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: "<html><body></body></html>")
        assert await WebSearchTool().run(query="q") == "No results found."

    async def test_garbage_html_is_graceful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: "<div><a>broken")
        assert await WebSearchTool().run(query="q") == "No results found."

    async def test_challenge_page_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: "anomaly challenge page")
        with pytest.raises(ToolError, match="blocked"):
            await WebSearchTool().run(query="q")

    async def test_title_only_result_still_emitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = (
            '<div class="result"><a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx">Solo</a></div>'
        )
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebSearchTool().run(query="q")
        assert "1. Solo" in result
        assert "https://example.com/x" in result

    async def test_network_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> str:
            raise ToolError("Web request failed: boom")

        monkeypatch.setattr(web_tools, "_http_get", boom)
        with pytest.raises(ToolError, match="Web request failed"):
            await WebSearchTool().run(query="q")

    async def test_empty_query_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: _ddg_html(["A"]))
        with pytest.raises(ToolError, match="must not be empty"):
            await WebSearchTool().run(query="   ")

    async def test_direct_href_kept_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = '<a class="result__a" href="https://direct.example/x">Direct</a>'
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebSearchTool().run(query="q")
        assert "https://direct.example/x" in result

    async def test_ddg_l_without_uddg_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = '<a class="result__a" href="//duckduckgo.com/l/?other=1">Odd</a>'
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebSearchTool().run(query="q")
        assert "https://duckduckgo.com/l/?other=1" in result


class TestHttpGetHelper:
    def test_200_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = SimpleNamespace(status_code=200, text="hello")
        monkeypatch.setattr(requests, "get", lambda *a, **k: fake)
        assert web_tools._http_get("https://example.com") == "hello"

    def test_non_200_raises_tool_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = SimpleNamespace(status_code=202, text="challenge")
        monkeypatch.setattr(requests, "get", lambda *a, **k: fake)
        with pytest.raises(ToolError, match="HTTP 202"):
            web_tools._http_get("https://example.com")

    def test_timeout_wraps_tool_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> SimpleNamespace:
            raise requests.Timeout("timed out")

        monkeypatch.setattr(requests, "get", boom)
        with pytest.raises(ToolError, match="Web request failed"):
            web_tools._http_get("https://example.com")

    def test_records_request_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_get(*args: object, **kwargs: object) -> SimpleNamespace:
            calls.append((args, kwargs))
            return SimpleNamespace(status_code=200, text="ok")

        monkeypatch.setattr(requests, "get", fake_get)
        web_tools._http_get("https://example.com", params={"q": "x"}, timeout=15.0)

        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] == "https://example.com"
        assert kwargs["params"] == {"q": "x"}
        assert kwargs["timeout"] == 15.0
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        assert "User-Agent" in headers


class TestWebFetchToolContract:
    def test_name(self) -> None:
        assert WebFetchTool().name == "web_fetch"

    def test_schema_requires_url(self) -> None:
        schema = WebFetchTool().input_schema
        assert schema["type"] == "object"
        assert schema["required"] == ["url"]
        assert "url" in schema["properties"]

    def test_schema_max_chars_bounds(self) -> None:
        max_chars = WebFetchTool().input_schema["properties"]["max_chars"]
        assert max_chars["minimum"] == 100
        assert max_chars["default"] == 2000


class TestWebFetchToolRun:
    async def test_extracts_readable_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = "<html><body><h1>Hi</h1><p>Hello world</p></body></html>"
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebFetchTool().run(url="https://example.com")
        assert result == "Hi Hello world"

    async def test_strips_script_and_style(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = "<script>var x=1;</script><style>.a{}</style><p>Keep me</p>"
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebFetchTool().run(url="https://example.com")
        assert result == "Keep me"

    async def test_truncates_long_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = f"<p>{'a' * 200}</p>"
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebFetchTool().run(url="https://example.com", max_chars=150)
        assert result.startswith("a" * 150 + "\n[truncated to first 150 chars]")
        assert len(result.splitlines()[0]) == 150

    async def test_default_truncation_on_long_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = f"<p>{'x' * 5000}</p>"
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebFetchTool().run(url="https://example.com")
        assert "[truncated to first 2000 chars]" in result

    async def test_max_chars_low_clamps_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = "<p>short</p>"
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebFetchTool().run(url="https://example.com", max_chars=50)
        assert result == "short"
        assert "[truncated" not in result

    async def test_rejects_ftp_scheme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: "unused")
        with pytest.raises(ToolError, match="only http/https"):
            await WebFetchTool().run(url="ftp://example.com/x")

    async def test_rejects_missing_scheme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: "unused")
        with pytest.raises(ToolError, match="only http/https"):
            await WebFetchTool().run(url="example.com/page")

    async def test_network_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> str:
            raise ToolError("Web request failed: boom")

        monkeypatch.setattr(web_tools, "_http_get", boom)
        with pytest.raises(ToolError, match="Web request failed"):
            await WebFetchTool().run(url="https://example.com")

    async def test_decodes_entities(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = "<p>AT&amp;T</p>"
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebFetchTool().run(url="https://example.com")
        assert result == "AT&T"

    async def test_skips_nested_script_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        html = "<script>if (a < b) {}</script><p>ok</p>"
        monkeypatch.setattr(web_tools, "_http_get", lambda *a, **k: html)
        result = await WebFetchTool().run(url="https://example.com")
        assert result == "ok"


class TestWebToolsRegistryIntegration:
    def test_registry_includes_web_tools(self) -> None:
        registry = ToolRegistry()
        assert "web_search" in registry.tool_names
        assert "web_fetch" in registry.tool_names

    def test_anthropic_defs_include_web_tools(self) -> None:
        defs = ToolRegistry().anthropic_tool_defs()
        by_name = {d["name"]: d for d in defs}
        for name in ("web_search", "web_fetch"):
            schema = by_name[name]["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
