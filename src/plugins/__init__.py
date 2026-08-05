"""Plugin discovery — loads external tools via ``importlib.metadata`` entry points.

Third-party packages can register tools by adding an entry point under the
``taskflow.tools`` group in their ``pyproject.toml``::

    [project.entry-points."taskflow.tools"]
    my_tool = "my_package.module:MyToolClass"

Each entry point may resolve to a :class:`Tool` subclass, a ``Tool``
instance, or a zero-argument callable returning one. The result must expose
``name``, ``description``, ``input_schema`` and an async ``run(**kwargs)``
method. Malformed, broken, or duplicate plugins are logged and skipped —
one bad plugin never blocks startup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.tools.base import Tool

# The four members the rest of the app requires of a tool.
_REQUIRED_ATTRS = ("name", "description", "input_schema", "run")


def discover_tools() -> list[Tool]:
    """Discover and instantiate all third-party tools registered via entry points.

    Each entry point must resolve to a :class:`Tool` subclass, a ``Tool``
    instance, or a zero-argument callable returning one, exposing ``name``,
    ``description``, ``input_schema``, and an async ``run`` method.

    Broken entry points are logged and skipped — one bad plugin never
    blocks the rest.
    """
    tools: list[Tool] = []
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return tools  # Python < 3.9 fallback

    try:
        eps = entry_points(group="taskflow.tools")
    except TypeError:
        # Some Python versions raise TypeError for unknown groups
        logger.debug("No taskflow.tools entry points found")
        return tools

    for ep in eps:
        try:
            raw = ep.load()
            if isinstance(raw, type):
                instance = raw()  # Tool subclass -> construct it
            elif callable(raw):
                instance = raw()  # factory callable -> call it
            else:
                instance = raw  # Tool instance loaded directly
            if all(hasattr(instance, attr) for attr in _REQUIRED_ATTRS):
                tools.append(instance)
                logger.info("Loaded plugin tool: %s from %s", instance.name, ep.module)
            else:
                logger.warning(
                    "Plugin %s from %s does not look like a Tool — skipping",
                    ep.name,
                    ep.module,
                )
        except Exception as exc:
            logger.error("Failed to load plugin %s: %s", ep.name, exc)

    logger.debug("Discovered %d plugin tool(s)", len(tools))
    return tools
