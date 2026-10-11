"""Send e-mail through SMTP (Gmail by default). Uses only the Python standard library."""
from __future__ import annotations

import re
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage

from src.config import settings

MAX_RECIPIENTS = 5
_EMAIL_RE = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")


class EmailError(Exception):
    """E-mail could not be sent. The message is safe to show (it never contains the password)."""


@dataclass(frozen=True)
class Attachment:
    filename: str
    data: bytes
    mime: str = "application/octet-stream"


@dataclass(frozen=True)
class SMTPConfig:
    host: str
    port: int
    user: str
    password: str
    recipients: list[str] = field(default_factory=list)

    @classmethod
    def from_settings(cls, recipients: list[str]) -> "SMTPConfig":
        return cls(settings.smtp_host, settings.smtp_port, settings.smtp_user, settings.smtp_password, recipients)


def parse_recipients(text: str) -> list[str]:
    """Split 'a@x.com, b@y.org' into a validated list."""
    parts = [p.strip() for p in re.split(r"[,;]", text or "") if p.strip()]
    if not parts:
        raise ValueError("Enter at least one recipient email address.")
    if len(parts) > MAX_RECIPIENTS:
        raise ValueError(f"At most {MAX_RECIPIENTS} recipients are allowed.")
    bad = [p for p in parts if not _EMAIL_RE.match(p)]
    if bad:
        raise ValueError("These do not look like email addresses: " + ", ".join(bad))
    return parts


def send_email(
    config: SMTPConfig,
    subject: str,
    text: str,
    html: str | None = None,
    attachments: list[Attachment] | None = None,
    timeout: int = 30,
) -> None:
    if not config.user or not config.password:
        raise EmailError("Email is not set up. Fill in SMTP_USER and SMTP_PASSWORD in .env, then restart the app.")
    if not config.recipients:
        raise EmailError("There is no recipient. Set ALERT_RECIPIENT in .env or type an address on the Alerts page.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config.user
    message["To"] = ", ".join(config.recipients)
    message.set_content(text)
    if html:
        message.add_alternative(html, subtype="html")
    for item in attachments or []:
        maintype, _, subtype = item.mime.partition("/")
        message.add_attachment(item.data, maintype=maintype, subtype=subtype or "octet-stream", filename=item.filename)

    password = config.password.replace(" ", "")  # Google shows app passwords in groups of four
    try:
        if config.port == 465:
            with smtplib.SMTP_SSL(
                config.host, config.port, timeout=timeout, context=ssl.create_default_context()
            ) as server:
                server.login(config.user, password)
                server.send_message(message)
        else:
            with smtplib.SMTP(config.host, config.port, timeout=timeout) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(config.user, password)
                server.send_message(message)
    except smtplib.SMTPAuthenticationError:
        raise EmailError(
            "The mail server rejected the login. Use a 16-character App Password (not your normal Gmail "
            "password), make sure 2-Step Verification is on, and check SMTP_USER. If you changed your Google "
            "password, create a new App Password."
        ) from None
    except smtplib.SMTPRecipientsRefused:
        raise EmailError("The mail server refused the recipient address. Check the address.") from None
    except smtplib.SMTPConnectError:
        raise EmailError(f"Could not connect to {config.host}:{config.port}. Check your internet connection.") from None
    except smtplib.SMTPException as exc:
        raise EmailError(f"The mail server rejected the message ({exc.__class__.__name__}).") from None
    except OSError:
        raise EmailError(
            f"Could not reach {config.host}:{config.port}. Check your internet connection. A firewall or "
            "antivirus may block port 587: try SMTP_PORT=465 in .env."
        ) from None