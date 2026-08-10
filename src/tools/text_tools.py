"""Text & encoding tools — base64, URL encoding, UUIDs, hashing, word counts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import uuid
from typing import Any
from urllib.parse import quote, unquote

from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)

#: Hashing algorithms exposed to the model. Keep the list fixed so arbitrary
#: ``hashlib`` names (and the modules behind them) are never reachable.
HASH_ALGORITHMS: frozenset[str] = frozenset({"md5", "sha1", "sha224", "sha256", "sha384", "sha512"})


class Base64EncodeTool(Tool):
    """Encode a string as base64."""

    @property
    def name(self) -> str:
        return "base64_encode"

    @property
    def description(self) -> str:
        return "Encode a string as base64 (UTF-8)."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to encode"},
            },
            "required": ["text"],
        }

    async def run(self, text: str, **kwargs: Any) -> str:  # type: ignore[override]
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return encoded


class Base64DecodeTool(Tool):
    """Decode a base64 string back to UTF-8 text."""

    @property
    def name(self) -> str:
        return "base64_decode"

    @property
    def description(self) -> str:
        return "Decode a base64 string back to UTF-8 text."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The base64 text to decode"},
            },
            "required": ["text"],
        }

    async def run(self, text: str, **kwargs: Any) -> str:  # type: ignore[override]
        padded = text.strip()
        padded += "=" * ((-len(padded)) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ToolError(f"Invalid base64 input: {exc}") from exc
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(
                "Decoded bytes are not valid UTF-8 text (the payload is binary, not text)."
            ) from exc


class UrlEncodeTool(Tool):
    """Percent-encode a string for use in a URL."""

    @property
    def name(self) -> str:
        return "url_encode"

    @property
    def description(self) -> str:
        return "Percent-encode a string for use in a URL."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The string to encode"},
            },
            "required": ["text"],
        }

    async def run(self, text: str, **kwargs: Any) -> str:  # type: ignore[override]
        return quote(text)


class UrlDecodeTool(Tool):
    """Decode a percent-encoded URL string."""

    @property
    def name(self) -> str:
        return "url_decode"

    @property
    def description(self) -> str:
        return "Decode a percent-encoded URL string back to plain text."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The encoded string to decode"},
            },
            "required": ["text"],
        }

    async def run(self, text: str, **kwargs: Any) -> str:  # type: ignore[override]
        return unquote(text)


class UuidTool(Tool):
    """Generate a random UUID (version 4)."""

    @property
    def name(self) -> str:
        return "uuid4"

    @property
    def description(self) -> str:
        return "Generate a random UUID (version 4) and return it as a string."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> str:
        return str(uuid.uuid4())


class HashTool(Tool):
    """Hash a string with a supported algorithm."""

    @property
    def name(self) -> str:
        return "hash_text"

    @property
    def description(self) -> str:
        algorithms = ", ".join(sorted(HASH_ALGORITHMS))
        return f"Compute the {algorithms} hash of a string (hex digest)."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to hash"},
                "algorithm": {
                    "type": "string",
                    "description": "Hash algorithm (default: sha256)",
                    "enum": sorted(HASH_ALGORITHMS),
                },
            },
            "required": ["text"],
        }

    async def run(self, text: str, **kwargs: Any) -> str:  # type: ignore[override]
        algorithm = kwargs.get("algorithm", "sha256")
        if not isinstance(algorithm, str) or algorithm not in HASH_ALGORITHMS:
            raise ToolError(f"Unsupported hash algorithm: {algorithm!r}")
        digest = hashlib.new(algorithm, text.encode("utf-8")).hexdigest()
        return digest


class WordCountTool(Tool):
    """Count words, characters, and lines in a string."""

    @property
    def name(self) -> str:
        return "word_count"

    @property
    def description(self) -> str:
        return "Count the words, characters, and lines in a string."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to count"},
            },
            "required": ["text"],
        }

    async def run(self, text: str, **kwargs: Any) -> str:  # type: ignore[override]
        words = len(text.split())
        chars = len(text)
        lines = len(text.splitlines())
        return f"{words} words · {chars} characters · {lines} lines"
