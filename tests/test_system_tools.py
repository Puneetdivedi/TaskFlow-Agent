"""Tests for the everyday helper tools (calculator, clipboard, system info)."""

from __future__ import annotations

import inspect

import pytest

from src.tools import system_tools
from src.tools.base import ToolError
from src.tools.system_tools import (
    CalculatorTool,
    ClipboardReadTool,
    ClipboardWriteTool,
    SystemInfoTool,
)


class TestCalculatorTool:
    def test_name(self) -> None:
        assert CalculatorTool().name == "calculator"

    async def test_arithmetic_precedence(self) -> None:
        assert await CalculatorTool().run("1 + 2 * 3") == "1 + 2 * 3 = 7"

    async def test_parentheses(self) -> None:
        assert await CalculatorTool().run("(1 + 2) * 3") == "(1 + 2) * 3 = 9"

    async def test_unary_minus(self) -> None:
        assert await CalculatorTool().run("-5 + 3") == "-5 + 3 = -2"

    async def test_division_and_modulo(self) -> None:
        assert await CalculatorTool().run("10 / 4") == "10 / 4 = 2.5"
        assert await CalculatorTool().run("10 % 3") == "10 % 3 = 1"

    async def test_pow(self) -> None:
        assert await CalculatorTool().run("2 ** 10") == "2 ** 10 = 1024"

    async def test_floor_div(self) -> None:
        assert await CalculatorTool().run("10 // 3") == "10 // 3 = 3"

    async def test_constant_pi(self) -> None:
        assert await CalculatorTool().run("pi") == "pi = 3.14159"

    async def test_sqrt(self) -> None:
        assert await CalculatorTool().run("sqrt(16)") == "sqrt(16) = 4"

    async def test_min_max_round(self) -> None:
        assert await CalculatorTool().run("min(3, 1, 2)") == "min(3, 1, 2) = 1"
        assert await CalculatorTool().run("max(3, 1, 2)") == "max(3, 1, 2) = 3"
        assert await CalculatorTool().run("round(2.567, 2)") == "round(2.567, 2) = 2.57"

    async def test_division_by_zero_raises(self) -> None:
        with pytest.raises(ToolError, match="Division by zero"):
            await CalculatorTool().run("1 / 0")

    async def test_import_blocked(self) -> None:
        with pytest.raises(ToolError, match="Unknown function"):
            await CalculatorTool().run("__import__('os')")

    async def test_attribute_access_blocked(self) -> None:
        with pytest.raises(ToolError, match="Unsupported expression element"):
            await CalculatorTool().run("a.b")

    async def test_unknown_name_blocked(self) -> None:
        with pytest.raises(ToolError, match="Unknown name"):
            await CalculatorTool().run("x + 1")

    async def test_multiple_statements_blocked(self) -> None:
        with pytest.raises(ToolError, match="Invalid expression"):
            await CalculatorTool().run("1; 2")

    async def test_result_too_large_raises(self) -> None:
        with pytest.raises(ToolError, match="Result too large"):
            await CalculatorTool().run("2 ** 100")

    async def test_float_overflow_raises(self) -> None:
        with pytest.raises(ToolError, match="Numeric overflow"):
            await CalculatorTool().run("10.0 ** 1000")

    def test_no_eval_or_exec_in_source(self) -> None:
        """The calculator must never fall back to eval/exec/compile."""
        source = inspect.getsource(system_tools)
        for banned in ("eval(", "exec(", "compile(", "__import__("):
            assert banned not in source


class _FakeClipboard:
    def __init__(self) -> None:
        self.content = ""

    def copy(self, text: str) -> None:
        self.content = text

    def paste(self) -> str:
        return self.content


class TestClipboardReadTool:
    def test_name(self) -> None:
        assert ClipboardReadTool().name == "clipboard_read"

    async def test_reads_clipboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeClipboard()
        fake.content = "hello"
        monkeypatch.setattr(system_tools, "_pyperclip", lambda: fake)
        assert await ClipboardReadTool().run() == "hello"

    async def test_empty_clipboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(system_tools, "_pyperclip", lambda: _FakeClipboard())
        assert await ClipboardReadTool().run() == "Clipboard is empty."


class TestClipboardWriteTool:
    def test_name(self) -> None:
        assert ClipboardWriteTool().name == "clipboard_write"

    async def test_writes_and_reports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeClipboard()
        monkeypatch.setattr(system_tools, "_pyperclip", lambda: fake)
        result = await ClipboardWriteTool().run("hello")
        assert fake.content == "hello"
        assert result == "Copied 5 characters to the clipboard."


class TestPyperclipGuard:
    def test_missing_pyperclip_raises_toolerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(name: str) -> object:
            raise ImportError(f"no module named {name}")

        monkeypatch.setattr(system_tools.importlib, "import_module", _boom)
        with pytest.raises(ToolError, match="pyperclip"):
            system_tools._pyperclip()


class TestSystemInfoTool:
    def test_name(self) -> None:
        assert SystemInfoTool().name == "system_info"

    async def test_includes_expected_fields(self) -> None:
        result = await SystemInfoTool().run()
        assert "System: " in result
        assert "Python: " in result
        assert "Working dir: " in result
        assert "Hostname: " in result
