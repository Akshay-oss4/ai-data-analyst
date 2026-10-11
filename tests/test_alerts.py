"""Tests for alert rules, duplicate protection and the e-mail sender (no real network)."""
import smtplib

import pytest

from src.alerts.alerting import (
    build_alert_message,
    fingerprint,
    process_alert,
    select_alert_findings,
)
from src.alerts.email_alert import Attachment, EmailError, SMTPConfig, parse_recipients, send_email
from src.analysis.findings import Finding
from src.database import alert_log


def make_findings():
    return [
        Finding("anomaly", "high", "A", "detail a"),
        Finding("anomaly", "medium", "B", "detail b"),
        Finding("trend", "high", "C", "detail c"),
        Finding("correlation", "info", "D", "detail d"),
        Finding("anomaly", "low", "E", "detail e"),
    ]


CONFIG = SMTPConfig("smtp.test", 587, "me@gmail.com", "abcd efgh ijkl mnop", ["you@example.com"])


def make_fake_smtp(login_error=None):
    class FakeSMTP:
        instances = []

        def __init__(self, host, port, timeout=None):
            self.calls = []
            self.message = None
            FakeSMTP.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self, context=None):
            self.calls.append("starttls")

        def login(self, user, password):
            if login_error:
                raise login_error
            self.calls.append(("login", user, password))

        def send_message(self, message):
            self.calls.append("send")
            self.message = message

    return FakeSMTP


def test_select_alert_findings():
    findings = make_findings()
    assert [f.title for f in select_alert_findings(findings, "high")] == ["A"]
    assert [f.title for f in select_alert_findings(findings, "medium")] == ["A", "B"]
    assert [f.title for f in select_alert_findings(findings, "high", include_trends=True)] == ["A", "C"]


def test_fingerprint_is_stable_and_specific():
    findings = make_findings()
    assert fingerprint(findings, "x.csv") == fingerprint(list(reversed(findings)), "x.csv")
    assert fingerprint(findings, "x.csv") != fingerprint(findings, "y.csv")
    assert fingerprint(findings, "x.csv") != fingerprint(findings[:2], "x.csv")


def test_alert_message_escapes_html_and_names_the_scope():
    findings = [Finding("anomaly", "high", "<b>bold</b>", "detail")]
    message = build_alert_message("sales.csv", findings)
    assert "1 alert(s) in sales.csv" in message.subject
    assert "<b>bold</b>" in message.text
    assert "&lt;b&gt;bold&lt;/b&gt;" in message.html
    assert "<b>bold</b>" not in message.html


def test_parse_recipients():
    assert parse_recipients("a@b.com, c@d.org") == ["a@b.com", "c@d.org"]
    for bad in ["", "not-an-email", "a@b.com, nope", "a@b.com,b@b.com,c@b.com,d@b.com,e@b.com,f@b.com"]:
        with pytest.raises(ValueError):
            parse_recipients(bad)


def test_send_email_builds_message_with_attachment(monkeypatch):
    fake = make_fake_smtp()
    monkeypatch.setattr(smtplib, "SMTP", fake)
    send_email(CONFIG, "Subject", "plain", "<p>html</p>", [Attachment("report.html", b"<html></html>", "text/html")])

    server = fake.instances[-1]
    assert "starttls" in server.calls
    assert ("login", "me@gmail.com", "abcdefghijklmnop") in server.calls  # spaces removed
    assert server.message["To"] == "you@example.com"
    assert server.message["Subject"] == "Subject"
    assert "report.html" in [a.get_filename() for a in server.message.iter_attachments()]


def test_send_email_login_failure_gives_helpful_message(monkeypatch):
    error = smtplib.SMTPAuthenticationError(535, b"Username and Password not accepted")
    monkeypatch.setattr(smtplib, "SMTP", make_fake_smtp(login_error=error))
    with pytest.raises(EmailError, match="App Password"):
        send_email(CONFIG, "Subject", "plain")


def test_alert_log_remembers_only_successful_sends(tmp_path):
    db = tmp_path / "alerts.db"
    alert_log.log_alert(db, "fp1", "x.csv", "high", 2, "a@b.com", "sent")
    alert_log.log_alert(db, "fp2", "x.csv", "high", 2, "a@b.com", "failed", "boom")
    assert alert_log.already_sent(db, "fp1")
    assert not alert_log.already_sent(db, "fp2")
    assert not alert_log.already_sent(db, "other")
    assert [r["status"] for r in alert_log.recent_alerts(db)] == ["failed", "sent"]  # newest first


def test_process_alert_sends_once_and_retries_after_failure(tmp_path):
    db = tmp_path / "alerts.db"
    findings = make_findings()
    sent_subjects = []

    def good_sender(config, subject, text, html, attachments):
        sent_subjects.append(subject)

    def bad_sender(config, subject, text, html, attachments):
        raise EmailError("boom")

    # Nothing qualifies: nothing is sent
    quiet = process_alert([findings[3]], "x.csv", CONFIG, db, sender=good_sender)
    assert quiet.status == "no_findings" and not sent_subjects

    # A failed send is logged but does not block a retry
    assert process_alert(findings, "x.csv", CONFIG, db, sender=bad_sender).status == "failed"
    assert process_alert(findings, "x.csv", CONFIG, db, sender=good_sender).status == "sent"

    # The same findings are not sent twice, unless forced
    assert process_alert(findings, "x.csv", CONFIG, db, sender=good_sender).status == "duplicate"
    assert process_alert(findings, "x.csv", CONFIG, db, sender=good_sender, force=True).status == "sent"
    assert len(sent_subjects) == 2