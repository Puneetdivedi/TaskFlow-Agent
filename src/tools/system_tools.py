"""Everyday helper tools — safe calculator, clipboard, and system info."""

from __future__ import annotations

import ast
import importlib
import logging
import math
import operator
import os
import platform
import socket
from pathlib import Path
from typing import Any, Callable

from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)

#: Name -> value constants the calculator may reference.
_CALC_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

#: Name -> callable functions the calculator may call (positional args only).
_CALC_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}

#: Binop AST node -> operator callable. Division is guarded separately.
_CALC_BINOPS: dict[type[ast.operator], Callable[[float, float], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_CALC_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_RESULT = 10**15


def _eval_calc(node: ast.AST) -> Any:
    """Evaluate a whitelisted arithmetic AST node by hand — never ``eval``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _CALC_BINOPS.get(type(node.op))
        if op is None:
            raise ToolError("Unsupported binary operator in expression")
        left = _eval_calc(node.left)
        right = _eval_calc(node.right)
        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ToolError("Division by zero is not allowed")
        try:
            return op(left, right)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ToolError("Numeric overflow in expression") from exc
    if isinstance(node, ast.UnaryOp):
        unary_op = _CALC_UNARY.get(type(node.op))
        if unary_op is None:
            raise ToolError("Unsupported unary operator in expression")
        return unary_op(_eval_calc(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CALC_CONSTANTS:
            return _CALC_CONSTANTS[node.id]
        raise ToolError(f"Unknown name in expression: {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ToolError("Only named function calls with positional args are supported")
        func = _CALC_FUNCS.get(node.func.id)
        if func is None:
            raise ToolError(f"Unknown function in expression: {node.func.id!r}")
        args = [_eval_calc(arg) for arg in node.args]
        try:
            return func(*args)
        except (OverflowError, ValueError, ZeroDivisionError, TypeError) as exc:
            raise ToolError(f"Function {node.func.id} failed: {exc}") from exc
    raise ToolError(f"Unsupported expression element: {type(node).__name__}")


def _format_result(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


class CalculatorTool(Tool):
    """Evaluate a safe arithmetic expression (no eval)."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a math expression. Supports + - * / // % **, parentheses, "
            "constants pi/e/tau, and functions sqrt, log, log10, exp, abs, round, "
            "min, max, floor, ceil. No variables or arbitrary code."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    async def run(self, expression: str, **kwargs: Any) -> str:  # type: ignore[override]
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"Invalid expression: {exc}") from exc
        try:
            result = _eval_calc(tree.body)
        except (OverflowError, RecursionError) as exc:
            raise ToolError("Expression overflowed or recursed too deeply") from exc
        if isinstance(result, (int, float)):
            if isinstance(result, int):
                if abs(result) > _MAX_RESULT:
                    raise ToolError("Result too large")
            elif not math.isfinite(result) or abs(result) > _MAX_RESULT:
                raise ToolError("Result too large")
        return f"{expression} = {_format_result(result)}"


def _pyperclip() -> Any:
    """Return the pyperclip module, or raise a friendly ``ToolError``."""
    try:
        return importlib.import_module("pyperclip")
    except ImportError as exc:
        raise ToolError("Clipboard access requires the 'pyperclip' package") from exc


class ClipboardReadTool(Tool):
    """Read the current clipboard contents."""

    @property
    def name(self) -> str:
        return "clipboard_read"

    @property
    def description(self) -> str:
        return "Return the current clipboard contents as text."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> str:
        text = str(_pyperclip().paste())
        if not text:
            return "Clipboard is empty."
        return text


class ClipboardWriteTool(Tool):
    """Copy text to the clipboard."""

    @property
    def name(self) -> str:
        return "clipboard_write"

    @property
    def description(self) -> str:
        return "Copy the given text to the clipboard."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to copy to the clipboard"},
            },
            "required": ["text"],
        }

    async def run(self, text: str, **kwargs: Any) -> str:  # type: ignore[override]
        _pyperclip().copy(text)
        return f"Copied {len(text)} characters to the clipboard."


class SystemInfoTool(Tool):
    """Return basic info about the machine the agent runs on."""

    @property
    def name(self) -> str:
        return "system_info"

    @property
    def description(self) -> str:
        return (
            "Return basic system information: OS, architecture, Python version, "
            "hostname, home directory, and working directory."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> str:
        return "\n".join(
            [
                f"System: {platform.system()} {platform.release()} ({platform.machine()})",
                f"OS name: {os.name}",
                f"Python: {platform.python_version()}",
                f"Hostname: {socket.gethostname()}",
                f"Home: {Path.home()}",
                f"Working dir: {os.getcwd()}",
            ]
        )
