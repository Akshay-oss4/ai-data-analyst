"""Alert rules: which findings deserve an e-mail, the message text, and duplicate protection."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from html import escape as esc
from pathlib import Path

from src.alerts.email_alert import Attachment, EmailError, SMTPConfig, send_email
from src.analysis.findings import Finding
from src.database import alert_log

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}
_COLORS = {"high": "#dc2626", "medium": "#ea580c", "low": "#ca8a04", "info": "#2563eb"}


@dataclass
class AlertMessage:
    subject: str
    text: str
    html: str


@dataclass
class AlertOutcome:
    status: str  # sent | duplicate | no_findings | failed
    message: str
    findings: list[Finding]
    subject: str = ""


def select_alert_findings(
    findings: list[Finding], min_severity: str = "high", include_trends: bool = False
) -> list[Finding]:
    """Anomalies at or above the threshold (and optionally declining trends)."""
    limit = SEVERITY_RANK[min_severity]
    categories = {"anomaly"} | ({"trend"} if include_trends else set())
    return [f for f in findings if f.category in categories and SEVERITY_RANK[f.severity] <= limit]


def fingerprint(findings: list[Finding], scope_label: str) -> str:
    """Short hash that identifies this exact set of findings for this data scope."""
    lines = sorted(f"{f.category}|{f.column or ''}|{f.title}" for f in findings)
    return hashlib.sha256("\n".join([scope_label, *lines]).encode("utf-8")).hexdigest()[:16]


def build_alert_message(scope_label: str, findings: list[Finding], when: datetime | None = None) -> AlertMessage:
    when = when or datetime.now()
    scope = " ".join(scope_label.split())  # no line breaks in a subject
    short_scope = scope if len(scope) <= 80 else scope[:77] + "..."
    subject = f"[Data Analyst] {len(findings)} alert(s) in {short_scope}"

    text_lines = [f"Alerts found in: {scope}", f"Generated: {when:%Y-%m-%d %H:%M}", ""]
    html_rows = []
    for f in findings:
        text_lines += [f"[{f.severity.upper()}] {f.title}", f.detail, ""]
        html_rows.append(
            f'<tr><td style="padding:8px;vertical-align:top;white-space:nowrap">'
            f'<span style="background:{_COLORS.get(f.severity, "#2563eb")};color:#fff;border-radius:10px;'
            f'padding:2px 8px;font-size:12px">{esc(f.severity)}</span></td>'
            f'<td style="padding:8px"><strong>{esc(f.title)}</strong><br>{esc(f.detail)}</td></tr>'
        )
    footer = "These are statistical signals, not proof of a problem. Please review them in the dashboard."
    text_lines.append(footer)

    html = (
        '<div style="font-family:Segoe UI,Arial,sans-serif;max-width:640px">'
        f"<h2>{len(findings)} alert(s)</h2>"
        f'<p style="color:#6b7280">Data: {esc(scope)}<br>Generated: {when:%Y-%m-%d %H:%M}</p>'
        f'<table style="border-collapse:collapse;width:100%">{"".join(html_rows)}</table>'
        f'<p style="color:#6b7280;font-size:13px">{esc(footer)}</p></div>'
    )
    return AlertMessage(subject, "\n".join(text_lines), html)


def process_alert(
    findings: list[Finding],
    scope_label: str,
    config: SMTPConfig,
    db_path: Path | str,
    *,
    min_severity: str = "high",
    include_trends: bool = False,
    attachments: list[Attachment] | None = None,
    force: bool = False,
    sender=send_email,
) -> AlertOutcome:
    """Select findings, skip duplicates, send, and log the result."""
    selected = select_alert_findings(findings, min_severity, include_trends)
    if not selected:
        return AlertOutcome("no_findings", "No findings reach the alert threshold, so nothing was sent.", [])

    fp = fingerprint(selected, scope_label)
    message = build_alert_message(scope_label, selected)
    if not force and alert_log.already_sent(db_path, fp):
        return AlertOutcome(
            "duplicate",
            "An alert for these exact findings was already sent, so no new email was sent.",
            selected,
            message.subject,
        )

    recipients = ", ".join(config.recipients)
    try:
        sender(config, message.subject, message.text, message.html, attachments or [])
    except EmailError as exc:
        alert_log.log_alert(db_path, fp, scope_label, min_severity, len(selected), recipients, "failed", str(exc))
        return AlertOutcome("failed", str(exc), selected, message.subject)

    alert_log.log_alert(db_path, fp, scope_label, min_severity, len(selected), recipients, "sent")
    return AlertOutcome("sent", f"Alert sent to {recipients}.", selected, message.subject)