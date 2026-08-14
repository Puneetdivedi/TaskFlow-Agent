"""Desktop and email notifications for reminders and autonomous runs.

A single :class:`Notifier` owns both delivery channels:

- **Desktop** toasts use ``winotify`` and only work on Windows. The package is
  imported lazily so it can be missing without breaking the app: any failure
  (missing package, non-Windows host, ``show()`` error) degrades to a status
  string and a log line — never a raise.
- **Email** uses the stdlib ``smtplib`` + ``email.message`` with optional
  STARTTLS and login. Misconfiguration and send failures raise
  :class:`ToolError` so the agent sees a clear, actionable message.
"""

from __future__ import annotations

import importlib
import logging
import smtplib
from email.message import EmailMessage
from typing import Any

from src.tools.base import ToolError

logger = logging.getLogger(__name__)

#: Subject/body lengths are capped for desktop toasts (Windows toast limits).
_DESKTOP_TITLE_MAX = 64
_DESKTOP_MSG_MAX = 256


class Notifier:
    """Delivers desktop toasts and email notifications.

    Config comes from :class:`~config.settings.Settings` via
    :meth:`Notifier.from_settings`; an empty ``smtp_host`` / missing recipient
    means email raises a friendly :class:`ToolError` on use.
    """

    def __init__(
        self,
        *,
        desktop_enabled: bool = True,
        smtp_host: str = "",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        smtp_from: str = "",
        smtp_starttls: bool = True,
        notify_email_to: str = "",
    ) -> None:
        self._desktop_enabled = desktop_enabled
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_user = smtp_user
        self._smtp_password = smtp_password
        self._smtp_from = smtp_from
        self._smtp_starttls = smtp_starttls
        self._notify_email_to = notify_email_to

    @classmethod
    def from_settings(cls, settings: Any) -> "Notifier":
        """Build a :class:`Notifier` from a ``Settings`` instance."""
        return cls(
            desktop_enabled=settings.notify_desktop,
            smtp_host=settings.smtp_host,
            smtp_port=settings.smtp_port,
            smtp_user=settings.smtp_user,
            smtp_password=settings.smtp_password,
            smtp_from=settings.smtp_from,
            smtp_starttls=settings.smtp_starttls,
            notify_email_to=settings.notify_email_to,
        )

    # ------------------------------------------------------------------
    @property
    def email_configured(self) -> bool:
        """True when SMTP and a default recipient are both configured."""
        return bool(self._smtp_host and self._notify_email_to)

    @property
    def notify_email_to(self) -> str:
        return self._notify_email_to

    # ------------------------------------------------------------------
    def desktop(self, title: str, message: str) -> str:
        """Show a Windows toast; returns a status string. Never raises."""
        if not self._desktop_enabled:
            return "(Desktop notifications disabled.)"
        try:
            winotify = importlib.import_module("winotify")
            notification = winotify.Notification(
                app_id="TaskFlow Agent",
                title=title[:_DESKTOP_TITLE_MAX],
                msg=message[:_DESKTOP_MSG_MAX],
                duration="short",
            )
            notification.show()
        except Exception as exc:  # noqa: BLE001 — degrade, never raise
            logger.info("Desktop notification unavailable: %s", exc)
            return f"(Desktop notification unavailable: {exc})"
        logger.info("Desktop notification shown: %s", title)
        return f"(Desktop notification shown: {title})"

    def email(self, to: str | None, subject: str, body: str) -> str:
        """Send *body* to *to* (or the configured default recipient) via SMTP.

        Raises :class:`ToolError` when email is not configured or the send
        fails, so the caller can surface a clear message.
        """
        recipient = (to or self._notify_email_to).strip()
        if not recipient:
            raise ToolError(
                "Email not configured — set NOTIFY_EMAIL_TO in .env (or pass an address)."
            )
        if not self._smtp_host:
            raise ToolError("Email not configured — set SMTP_HOST in .env.")
        from_addr = self._smtp_from or self._smtp_user or recipient
        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)
        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=15) as server:
                if self._smtp_starttls:
                    server.starttls()
                if self._smtp_user:
                    server.login(self._smtp_user, self._smtp_password)
                server.send_message(msg)
        except Exception as exc:  # noqa: BLE001 — surface SMTP failures clearly
            raise ToolError(f"Email send failed: {exc}") from exc
        logger.info("Email sent to %s: %s", recipient, subject)
        return f"(Email sent to {recipient}: {subject})"
