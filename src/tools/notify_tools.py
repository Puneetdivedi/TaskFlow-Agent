"""Tools that deliver notifications — desktop toasts and email."""

from __future__ import annotations

from typing import Any

from src.notify import Notifier
from src.tools.base import Tool


class NotifyTool(Tool):
    """Show a desktop notification (Windows toast) to the user."""

    def __init__(self, notifier: Notifier) -> None:
        self._notifier = notifier

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "notify"

    @property
    def description(self) -> str:
        return (
            "Show a desktop notification (Windows toast) to the user. Use it to "
            "surface short alerts, reminders, or completion notices without "
            "interrupting the conversation."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short notification title (kept under ~64 chars).",
                },
                "message": {
                    "type": "string",
                    "description": "Notification body text (kept under ~256 chars).",
                },
            },
            "required": ["title", "message"],
        }

    # ------------------------------------------------------------------
    async def run(self, title: str, message: str, **kwargs: Any) -> str:  # type: ignore[override]
        return self._notifier.desktop(title, message)


class SendEmailTool(Tool):
    """Send an email via the configured SMTP server."""

    def __init__(self, notifier: Notifier) -> None:
        self._notifier = notifier

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return (
            "Send an email to a recipient via the configured SMTP server. Raises "
            "a clear error when SMTP is not configured (see SMTP_* in .env)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Email body text."},
            },
            "required": ["to", "subject", "body"],
        }

    # ------------------------------------------------------------------
    async def run(self, to: str, subject: str, body: str, **kwargs: Any) -> str:  # type: ignore[override]
        return self._notifier.email(to, subject, body)
