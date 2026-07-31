"""Centralised logging configuration for the TaskFlow agent.

Usage at application startup (never in library code)::

    from src.logging_config import configure_logging
    configure_logging()

All modules should use the standard ``logging.getLogger(__name__)``
pattern — no module should import or call ``configure_logging`` itself.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def configure_logging() -> None:
    """Configure the root logger with a structured text format.

    Settings are read from environment variables (which may be populated
    from a ``.env`` file by the settings layer):

        ``AGENT_LOG_LEVEL``
            One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` (default ``INFO``).
        ``AGENT_LOG_FILE``
            Optional path to a log file.  When omitted, logs go to stderr.

    The format includes: timestamp (ISO-8601), logger name, level, and
    the formatted message — machine- and human-friendly.
    """
    level_name = os.getenv("AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = (
        "%(asctime)s  %(name)-35s %(levelname)-8s %(message)s"
    )
    datefmt = "%Y-%m-%dT%H:%M:%S%z"

    handler: logging.Handler
    log_file = os.getenv("AGENT_LOG_FILE")
    if log_file:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(path), encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if configure_logging is called more than once.
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers[0] = handler
