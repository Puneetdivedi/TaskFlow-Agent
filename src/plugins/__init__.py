"""Plugin discovery — loads external tools via ``importlib.metadata`` entry points.

Third-party packages can register tools by adding an entry point under the
``taskflow.tools`` group in their ``pyproject.toml``::

    [project.entry-points."taskflow.tools"]
    my_tool = "my_package.module:MyToolClass"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.tools.base import Tool


def discover_tools() -> list[Tool]:
    """Discover and instantiate all third-party tools registered via entry points.

    Each entry point must resolve to a :class:`Tool` subclass (or any
    callable that returns a :class:`Tool`-like object with ``name``,
    ``description``, ``input_schema``, and an async ``run`` method).

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
            cls = ep.load()
            instance = cls() if isinstance(cls, type) else cls
            if hasattr(instance, "run") and hasattr(instance, "name"):
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
