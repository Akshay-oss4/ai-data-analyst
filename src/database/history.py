"""Analysis history stored in SQLite.

All SQL lives in HistoryStore, so a PostgreSQL version can later implement the same methods
without touching the rest of the app.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ai.insights import InsightsResult
from src.analysis.pipeline import AnalysisBundle
from src.cleaning.cleaner import CleaningResult
from src.config import settings

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    sheet TEXT,
    scope TEXT NOT NULL,
    filters TEXT NOT NULL DEFAULT '',
    n_rows INTEGER NOT NULL,
    n_columns INTEGER NOT NULL,
    n_rows_original INTEGER NOT NULL,
    n_findings INTEGER NOT NULL,
    n_high INTEGER NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    summary_source TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS run_kpis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    column_name TEXT,
    display TEXT NOT NULL,
    value_num REAL
);
CREATE TABLE IF NOT EXISTS run_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    detail TEXT NOT NULL,
    column_name TEXT
);
CREATE TABLE IF NOT EXISTS run_trends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    metric TEXT NOT NULL,
    assessment TEXT NOT NULL,
    change_pct REAL,
    last_period_change_pct REAL,
    n_periods INTEGER NOT NULL,
    period TEXT NOT NULL,
    aggregation TEXT NOT NULL,
    p_value REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_filename ON analysis_runs(filename);
CREATE INDEX IF NOT EXISTS idx_kpis_run ON run_kpis(run_id);
CREATE INDEX IF NOT EXISTS idx_findings_run ON run_findings(run_id);
CREATE INDEX IF NOT EXISTS idx_trends_run ON run_trends(run_id);
"""


# ---------------------------------------------------------------- plain data holders
@dataclass
class KPIRow:
    name: str
    kind: str
    column: str | None
    display: str
    value: float | None


@dataclass
class FindingRow:
    severity: str
    category: str
    title: str
    detail: str
    column: str | None = None


@dataclass
class TrendRow:
    metric: str
    assessment: str
    change_pct: float | None
    last_period_change_pct: float | None
    n_periods: int
    period: str
    aggregation: str
    p_value: float | None


@dataclass
class RunRecord:
    filename: str
    sheet: str | None
    scope: str
    filters: str
    rows: int
    columns: int
    rows_original: int
    summary: str
    summary_source: str
    kpis: list[KPIRow] = field(default_factory=list)
    findings: list[FindingRow] = field(default_factory=list)
    trends: list[TrendRow] = field(default_factory=list)


@dataclass
class RunComparison:
    run_a: dict
    run_b: dict
    kpis: pd.DataFrame
    only_in_a: list[str]
    only_in_b: list[str]
    in_both: list[str]


# ---------------------------------------------------------------- building a record
def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_run_record(
    filename: str,
    sheet: str | None,
    scope_label: str,
    filters_applied: list[str],
    cleaning: CleaningResult,
    bundle: AnalysisBundle,
    insights: InsightsResult,
) -> RunRecord:
    rows, cols = cleaning.after_shape
    return RunRecord(
        filename=filename,
        sheet=sheet,
        scope=scope_label,
        filters="; ".join(filters_applied),
        rows=rows,
        columns=cols,
        rows_original=cleaning.before_shape[0],
        summary=insights.summary,
        summary_source=f"llm: {insights.model}" if insights.source == "llm" else "rules",
        kpis=[KPIRow(k.name, k.kind, k.column, k.display, _number(k.value)) for k in bundle.kpis],
        findings=[FindingRow(f.severity, f.category, f.title, f.detail, f.column) for f in bundle.findings],
        trends=[
            TrendRow(t.metric, t.assessment, t.change_pct, t.last_period_change_pct,
                     t.n_periods, t.period, t.aggregation, t.p_value)
            for t in bundle.trends
        ],
    )


def run_fingerprint(record: RunRecord) -> str:
    """Identifies a run by its content, so saving the same analysis twice is a no-op."""
    parts = [
        record.scope,
        str(record.rows),
        *sorted(f"{k.name}={k.display}" for k in record.kpis),
        *sorted(f"{f.severity}|{f.title}" for f in record.findings),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _normalize_title(title: str) -> str:
    """'8 unusual values in revenue' and '9 unusual values in revenue' are the same finding."""
    return re.sub(r"\d[\d,.]*", "#", title)


# ---------------------------------------------------------------- the store
class HistoryStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # makes ON DELETE CASCADE work
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return conn

    def save_run(self, record: RunRecord) -> tuple[int, bool]:
        """Save a run. Returns (run_id, created). created is False if an identical run exists."""
        fingerprint = run_fingerprint(record)
        with closing(self._connect()) as conn:
            existing = conn.execute(
                "SELECT id FROM analysis_runs WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if existing:
                return int(existing["id"]), False
            with conn:  # one transaction for the run and all its rows
                cursor = conn.execute(
                    "INSERT INTO analysis_runs (created_at, fingerprint, filename, sheet, scope, filters, "
                    "n_rows, n_columns, n_rows_original, n_findings, n_high, summary, summary_source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now().isoformat(timespec="seconds"), fingerprint, record.filename,
                        record.sheet, record.scope, record.filters, record.rows, record.columns,
                        record.rows_original, len(record.findings),
                        sum(f.severity == "high" for f in record.findings),
                        record.summary, record.summary_source,
                    ),
                )
                run_id = int(cursor.lastrowid)
                conn.executemany(
                    "INSERT INTO run_kpis (run_id, name, kind, column_name, display, value_num) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [(run_id, k.name, k.kind, k.column, k.display, k.value) for k in record.kpis],
                )
                conn.executemany(
                    "INSERT INTO run_findings (run_id, ordinal, severity, category, title, detail, column_name) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (run_id, i, f.severity, f.category, f.title, f.detail, f.column)
                        for i, f in enumerate(record.findings, start=1)
                    ],
                )
                conn.executemany(
                    "INSERT INTO run_trends (run_id, metric, assessment, change_pct, last_period_change_pct, "
                    "n_periods, period, aggregation, p_value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (run_id, t.metric, t.assessment, t.change_pct, t.last_period_change_pct,
                         t.n_periods, t.period, t.aggregation, t.p_value)
                        for t in record.trends
                    ],
                )
        return run_id, True

    def list_runs(self, limit: int = 200) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, created_at, filename, sheet, scope, filters, n_rows, n_columns, "
                "n_findings, n_high, summary_source FROM analysis_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> dict | None:
        with closing(self._connect()) as conn:
            run = conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            data = dict(run)
            data["kpis"] = [
                dict(r) for r in conn.execute(
                    "SELECT name, kind, column_name, display, value_num FROM run_kpis "
                    "WHERE run_id = ? ORDER BY id", (run_id,))
            ]
            data["findings"] = [
                dict(r) for r in conn.execute(
                    "SELECT ordinal, severity, category, title, detail, column_name FROM run_findings "
                    "WHERE run_id = ? ORDER BY ordinal", (run_id,))
            ]
            data["trends"] = [
                dict(r) for r in conn.execute(
                    "SELECT metric, assessment, change_pct, last_period_change_pct, n_periods, period, "
                    "aggregation, p_value FROM run_trends WHERE run_id = ? ORDER BY id", (run_id,))
            ]
        return data

    def delete_run(self, run_id: int) -> bool:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute("DELETE FROM analysis_runs WHERE id = ?", (run_id,))
            return cursor.rowcount > 0

    def filenames(self) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT DISTINCT filename FROM analysis_runs ORDER BY filename").fetchall()
        return [r["filename"] for r in rows]

    def kpi_names(self, filename: str) -> list[str]:
        """Names of KPIs that have a number (so they can be charted) for this file."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT k.name FROM run_kpis k JOIN analysis_runs r ON r.id = k.run_id "
                "WHERE r.filename = ? AND k.value_num IS NOT NULL ORDER BY k.name", (filename,)
            ).fetchall()
        return [r["name"] for r in rows]

    def kpi_history(self, filename: str, kpi_name: str, only_unfiltered: bool = True) -> pd.DataFrame:
        """One KPI across the saved runs of a file, oldest first."""
        sql = (
            "SELECT r.id AS run_id, r.created_at, r.scope, k.value_num AS value "
            "FROM run_kpis k JOIN analysis_runs r ON r.id = k.run_id "
            "WHERE r.filename = ? AND k.name = ? AND k.value_num IS NOT NULL"
        )
        if only_unfiltered:
            sql += " AND r.filters = ''"
        sql += " ORDER BY r.id"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, (filename, kpi_name)).fetchall()
        frame = pd.DataFrame([dict(r) for r in rows], columns=["run_id", "created_at", "scope", "value"])
        frame["created_at"] = pd.to_datetime(frame["created_at"])
        return frame

    def compare(self, id_a: int, id_b: int) -> RunComparison | None:
        run_a, run_b = self.get_run(id_a), self.get_run(id_b)
        if run_a is None or run_b is None:
            return None

        kpis_a = {k["name"]: k for k in run_a["kpis"]}
        kpis_b = {k["name"]: k for k in run_b["kpis"]}
        rows = []
        for name in list(kpis_a) + [n for n in kpis_b if n not in kpis_a]:
            ka, kb = kpis_a.get(name), kpis_b.get(name)
            va = ka["value_num"] if ka else None
            vb = kb["value_num"] if kb else None
            change = pct = None
            if va is not None and vb is not None:
                change = round(vb - va, 4)
                pct = round((vb - va) / abs(va) * 100, 1) if va != 0 else None
            rows.append(
                {
                    "KPI": name,
                    "Run A": ka["display"] if ka else "",
                    "Run B": kb["display"] if kb else "",
                    "Change": change,
                    "Change %": pct,
                }
            )

        titles_a = {_normalize_title(f["title"]): f["title"] for f in run_a["findings"]}
        titles_b = {_normalize_title(f["title"]): f["title"] for f in run_b["findings"]}
        return RunComparison(
            run_a={k: run_a[k] for k in ("id", "created_at", "filename", "scope", "n_rows")},
            run_b={k: run_b[k] for k in ("id", "created_at", "filename", "scope", "n_rows")},
            kpis=pd.DataFrame(rows),
            only_in_a=[t for key, t in titles_a.items() if key not in titles_b],
            only_in_b=[t for key, t in titles_b.items() if key not in titles_a],
            in_both=[t for key, t in titles_b.items() if key in titles_a],
        )


def get_store() -> HistoryStore:
    """The one place that decides which database is used (SQLite today, PostgreSQL later)."""
    return HistoryStore(settings.database_path)