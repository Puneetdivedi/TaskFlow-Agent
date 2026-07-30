"""Application configuration — environment and settings management.

Typical entry point::

    from config.settings import init_settings
    settings = init_settings()
"""

from __future__ import annotations

from config.settings import Settings, init_settings

__all__ = [
    "Settings",
    "init_settings",
]
