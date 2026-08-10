"""Tests for the text & encoding tools."""

from __future__ import annotations

import re

import pytest

from src.tools.base import ToolError
from src.tools.text_tools import (
    Base64DecodeTool,
    Base64EncodeTool,
    HashTool,
    UrlDecodeTool,
    UrlEncodeTool,
    UuidTool,
    WordCountTool,
)

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


class TestBase64Encode:
    def test_name(self) -> None:
        assert Base64EncodeTool().name == "base64_encode"

    async def test_encodes_utf8(self) -> None:
        assert await Base64EncodeTool().run("hello world") == "aGVsbG8gd29ybGQ="

    async def test_encodes_unicode(self) -> None:
        assert await Base64EncodeTool().run("héllo") == "aMOpbGxv"


class TestBase64Decode:
    def test_name(self) -> None:
        assert Base64DecodeTool().name == "base64_decode"

    async def test_decodes_roundtrip(self) -> None:
        tool = Base64DecodeTool()
        encoded = await Base64EncodeTool().run("roundtrip ✓")
        assert await tool.run(encoded) == "roundtrip ✓"

    async def test_invalid_base64_raises(self) -> None:
        with pytest.raises(ToolError, match="Invalid base64"):
            await Base64DecodeTool().run("not!@base64")

    async def test_binary_payload_raises(self) -> None:
        # Valid base64 that decodes to non-UTF-8 bytes (0xFF 0xFE).
        with pytest.raises(ToolError, match="not valid UTF-8"):
            await Base64DecodeTool().run("//4=")


class TestUrlEncodeDecode:
    async def test_encode(self) -> None:
        assert await UrlEncodeTool().run("a b&c=d") == "a%20b%26c%3Dd"

    async def test_decode_roundtrip(self) -> None:
        tool = UrlDecodeTool()
        encoded = await UrlEncodeTool().run("hello world?/x")
        assert await tool.run(encoded) == "hello world?/x"

    async def test_decode_percent(self) -> None:
        assert await UrlDecodeTool().run("caf%C3%A9") == "café"


class TestUuid4:
    def test_name(self) -> None:
        assert UuidTool().name == "uuid4"

    async def test_returns_valid_v4_uuid(self) -> None:
        assert UUID_RE.match(await UuidTool().run()) is not None

    async def test_uuids_differ(self) -> None:
        tool = UuidTool()
        assert await tool.run() != await tool.run()


class TestHashText:
    def test_name(self) -> None:
        assert HashTool().name == "hash_text"

    async def test_sha256_known_vector(self) -> None:
        assert await HashTool().run("abc") == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    async def test_md5_known_vector(self) -> None:
        assert await HashTool().run("abc", algorithm="md5") == "900150983cd24fb0d6963f7d28e17f72"

    async def test_sha1_known_vector(self) -> None:
        assert await HashTool().run("abc", algorithm="sha1") == (
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        )

    async def test_algorithm_matches_schema_enum(self) -> None:
        enum = HashTool().input_schema["properties"]["algorithm"]["enum"]
        assert set(enum) == {"md5", "sha1", "sha224", "sha256", "sha384", "sha512"}

    async def test_unknown_algorithm_raises(self) -> None:
        with pytest.raises(ToolError, match="Unsupported hash algorithm"):
            await HashTool().run("abc", algorithm="sha3_256")


class TestWordCount:
    def test_name(self) -> None:
        assert WordCountTool().name == "word_count"

    async def test_counts(self) -> None:
        assert await WordCountTool().run("one two  three") == "3 words · 14 characters · 1 lines"

    async def test_counts_multiline(self) -> None:
        assert await WordCountTool().run("a\nb c\n") == "3 words · 6 characters · 2 lines"

    async def test_empty(self) -> None:
        assert await WordCountTool().run("") == "0 words · 0 characters · 0 lines"
