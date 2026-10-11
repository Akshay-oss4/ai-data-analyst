"""SQLite log of alert emails. Used to avoid sending the same alert twice."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alert_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    scope TEXT NOT NULL,
    min_severity TEXT NOT NULL,
    n_findings INTEGER NOT NULL,
    recipients TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT ''
)
"""


def _connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def log_alert(
    path: Path | str,
    fingerprint: str,
    scope: str,
    min_severity: str,
    n_findings: int,
    recipients: str,
    status: str,
    error: str = "",
) -> None:
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO alert_log (created_at, fingerprint, scope, min_severity, n_findings, recipients, status, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), fingerprint, scope, min_severity,
             n_findings, recipients, status, error),
        )


def already_sent(path: Path | str, fingerprint: str) -> bool:
    """True if an alert with this fingerprint was sent successfully before."""
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT 1 FROM alert_log WHERE fingerprint = ? AND status = 'sent' LIMIT 1", (fingerprint,)
        ).fetchone()
    return row is not None


def recent_alerts(path: Path | str, limit: int = 20) -> list[dict]:
    with closing(_connect(path)) as conn:
        cursor = conn.execute(
            "SELECT created_at, scope, n_findings, recipients, status, error "
            "FROM alert_log ORDER BY id DESC LIMIT ?", (limit,)
        )
        names = [c[0] for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]