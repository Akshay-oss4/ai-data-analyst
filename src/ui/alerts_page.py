"""The Alerts page, plus the optional automatic alert."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai.insights import generate_insights
from src.alerts.alerting import AlertOutcome, fingerprint, process_alert, select_alert_findings
from src.alerts.email_alert import Attachment, EmailError, SMTPConfig, parse_recipients, send_email
from src.config import settings
from src.database.alert_log import recent_alerts
from src.reporting.html_report import build_html_report
from src.ui.pages import AppContext
from src.ui.presenters import SEVERITY_ICONS

SEVERITY_CHOICES = {"high": "High only", "medium": "Medium and High"}


def _report_attachment(ctx: AppContext) -> Attachment:
    """The HTML report without charts: small, and contains no scripts."""
    stored = st.session_state.get("insights")
    if stored and stored[0] == ctx.insights_key:
        insights = stored[1]
    else:
        insights = generate_insights(ctx.bundle, ctx.cleaning, ctx.scope_label, use_llm=False)
    html = build_html_report(
        title=f"Data analysis report: {ctx.filename}",
        scope_label=ctx.scope_label,
        cleaning=ctx.cleaning,
        full_cleaning=ctx.full_cleaning,
        raw_profiles=ctx.raw_profiles,
        bundle=ctx.bundle,
        insights=insights,
        filters_applied=ctx.filters_applied,
        chart_mode="none",
    )
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(ctx.filename).stem) or "report"
    return Attachment(f"{stem}_report.html", html.encode("utf-8"), "text/html")


def _show_outcome(outcome: AlertOutcome) -> None:
    if outcome.status == "sent":
        st.success(outcome.message)
    elif outcome.status == "failed":
        st.error(outcome.message)
    else:
        st.info(outcome.message)


def run_auto_alert(ctx: AppContext) -> AlertOutcome | None:
    """Send an alert automatically, at most once per set of findings per session."""
    cfg = st.session_state.get("alert_cfg")
    if not cfg or not cfg.get("auto") or not (settings.smtp_user and settings.smtp_password):
        return None
    selected = select_alert_findings(ctx.bundle.findings, cfg["min_severity"], cfg["include_trends"])
    if not selected:
        return None

    attempted = st.session_state.setdefault("auto_alert_attempted", set())
    fp = fingerprint(selected, ctx.scope_label)
    if fp in attempted:
        return None
    attempted.add(fp)

    try:
        config = SMTPConfig.from_settings(parse_recipients(cfg["recipient"]))
    except ValueError:
        return None
    attachments = [_report_attachment(ctx)] if cfg["attach"] else []
    outcome = process_alert(
        ctx.bundle.findings, ctx.scope_label, config, settings.database_path,
        min_severity=cfg["min_severity"], include_trends=cfg["include_trends"], attachments=attachments,
    )
    st.session_state["last_alert_outcome"] = outcome
    return outcome


def render_alerts_page(ctx: AppContext) -> None:
    st.header("Alerts")
    st.markdown(
        "Get an email when the analysis finds important anomalies. Alerts use your own Gmail account "
        "(free). Nothing is sent unless you click a button below or switch on automatic alerts."
    )

    configured = bool(settings.smtp_user and settings.smtp_password)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sender", settings.smtp_user or "not set")
    c2.metric("App Password", "set" if settings.smtp_password else "not set")
    c3.metric("Default recipient", settings.alert_recipient or "not set")
    if not configured:
        st.warning("Email is not set up yet. Follow the steps below, then restart the app.")

    with st.expander("How to set up Gmail (free)", expanded=not configured):
        st.markdown(
            "1. Use a **personal @gmail.com** account. Work and school accounts often cannot create App Passwords.\n"
            "2. Turn on **2-Step Verification** (Google Account → Security).\n"
            "3. Open `myaccount.google.com/apppasswords`, create a password named `data-analyst`, "
            "and copy the 16 characters.\n"
            "4. In the project's `.env` file set `SMTP_USER`, `SMTP_PASSWORD` (the App Password) and "
            "`ALERT_RECIPIENT`, then **restart Streamlit**.\n\n"
            "Never share the App Password. If you change your Google password, create a new one."
        )

    st.subheader("Alert rules")
    min_severity = st.radio(
        "Alert me about anomalies that are", list(SEVERITY_CHOICES), format_func=SEVERITY_CHOICES.get,
        horizontal=True, key="alert_min_severity",
    )
    include_trends = st.checkbox("Also alert on sharply declining metrics", key="alert_trends")
    attach = st.checkbox("Attach the HTML report (without charts, small)", value=True, key="alert_attach")
    recipient_text = st.text_input(
        "Send alerts to (up to 5 addresses, separated by commas)",
        value=settings.alert_recipient or settings.smtp_user, key="alert_recipient",
    )
    auto = st.checkbox(
        "Send automatically when I analyze a file and qualifying findings are found",
        key="alert_auto",
        help="At most one email per set of findings. Takes effect after you have opened this page.",
    )
    # Widgets on other pages are forgotten by Streamlit, so keep the settings in a plain dict
    st.session_state["alert_cfg"] = {
        "auto": auto, "min_severity": min_severity, "include_trends": include_trends,
        "attach": attach, "recipient": recipient_text,
    }

    selected = select_alert_findings(ctx.bundle.findings, min_severity, include_trends)
    st.subheader("What would be sent")
    if ctx.filters_applied:
        st.caption("Based on the filtered view: " + "; ".join(ctx.filters_applied))
    if selected:
        for f in selected:
            st.markdown(f"{SEVERITY_ICONS[f.severity]} **{f.title}**  \n{f.detail}")
    else:
        st.info("No findings reach this threshold right now, so no alert would be sent.")

    try:
        config = SMTPConfig.from_settings(parse_recipients(recipient_text))
    except ValueError as exc:
        st.error(str(exc))
        return

    b1, b2 = st.columns(2)
    if b1.button("Send test email", key="alert_test"):
        try:
            send_email(
                config,
                "[Data Analyst] Test email",
                "This is a test email from the AI-Powered Automated Data Analyst. Your email setup works.",
            )
            st.success(f"Test email sent to {', '.join(config.recipients)}. Check your inbox (and spam folder).")
        except EmailError as exc:
            st.error(str(exc))

    if b2.button("Send alert now", key="alert_send", disabled=not selected):
        with st.spinner("Sending..."):
            attachments = [_report_attachment(ctx)] if attach else []
            st.session_state["last_alert_outcome"] = process_alert(
                ctx.bundle.findings, ctx.scope_label, config, settings.database_path,
                min_severity=min_severity, include_trends=include_trends, attachments=attachments,
            )

    outcome = st.session_state.get("last_alert_outcome")
    if outcome is not None:
        _show_outcome(outcome)
        if outcome.status == "duplicate" and st.button("Send anyway", key="alert_force"):
            attachments = [_report_attachment(ctx)] if attach else []
            st.session_state["last_alert_outcome"] = process_alert(
                ctx.bundle.findings, ctx.scope_label, config, settings.database_path,
                min_severity=min_severity, include_trends=include_trends, attachments=attachments, force=True,
            )
            st.rerun()

    st.subheader("Alert history")
    history = recent_alerts(settings.database_path)
    if history:
        st.dataframe(pd.DataFrame(history))
    else:
        st.caption("No alerts have been sent or attempted yet.")