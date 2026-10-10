"""Heuristics that choose the most important metrics and date column automatically."""
from __future__ import annotations

import re

from src.profiling.type_detector import DATETIME, NUMERIC, ColumnProfile

# Columns that should be averaged, never summed (prices, ratings, rates, ages, ...)
_NON_ADDITIVE_RE = re.compile(
    r"price|rating|score|avg|average|mean|ratio|pct|percent|margin|discount|rank|year|"
    r"temperature|latitude|longitude|duration|(?:^|[_\s\-])(?:age|rate)(?:$|[_\s\-])",
    re.IGNORECASE,
)
# Columns that make sense to add up (revenue, units, cost, clicks, ...)
_ADDITIVE_RE = re.compile(
    r"revenue|sales|amount|total|cost|profit|income|spend|expense|quantity|qty|units|orders|"
    r"clicks|views|impressions|visits|sessions|volume|count|payment|salary|budget|value",
    re.IGNORECASE,
)
_STRONG_METRIC_RE = re.compile(r"revenue|sales|profit|income|amount|value", re.IGNORECASE)
_DATE_NAME_RE = re.compile(r"date|time|day|month|created|order|timestamp|period", re.IGNORECASE)


def is_additive(name: str) -> bool:
    """True if summing this column is meaningful (revenue, units), False for prices/ratings."""
    n = str(name)
    if "total" in n.lower():
        return True
    if _NON_ADDITIVE_RE.search(n):
        return False
    return bool(_ADDITIVE_RE.search(n))


def pick_primary_metrics(profiles: list[ColumnProfile], k: int = 3) -> list[str]:
    """Rank numeric columns by how likely they are to be the 'business metrics'."""
    numeric = [p for p in profiles if p.inferred_type == NUMERIC]

    def score(p: ColumnProfile) -> int:
        s = 0
        if is_additive(p.name):
            s += 2
        if _STRONG_METRIC_RE.search(p.name):
            s += 1
        return s

    ranked = sorted(numeric, key=score, reverse=True)  # stable: keeps column order on ties
    return [p.name for p in ranked[:k]]


def pick_primary_date_column(profiles: list[ColumnProfile]) -> str | None:
    """Choose the most likely 'main' date column, or None if there isn't one."""
    candidates = [p for p in profiles if p.inferred_type == DATETIME and p.unique_count >= 2]
    if not candidates:
        return None
    best = max(candidates, key=lambda p: (bool(_DATE_NAME_RE.search(p.name)), p.unique_count))
    return best.name