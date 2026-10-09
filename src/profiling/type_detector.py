"""Automatic column type detection and profiling.

Detected types:
  numeric, categorical, datetime, boolean, id, text, constant, empty
Also detects numbers / dates / booleans that are stored as text (e.g. "$1,200", "2024-03-01").
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"
BOOLEAN = "boolean"
ID = "id"
TEXT = "text"
CONSTANT = "constant"
EMPTY = "empty"

BOOL_MAP = {
    "true": True, "yes": True, "y": True, "t": True, "1": True,
    "false": False, "no": False, "n": False, "f": False, "0": False,
}

_CURRENCY_RE = re.compile(r"[$€£¥₹]|\b(?:USD|EUR|GBP|INR)\b", re.IGNORECASE)
_ID_WORD_RE = re.compile(
    r"(?:^|[_\s\-])(?:id|uuid|guid|key|ref|sku|code|number|nr)(?:$|[_\s\-])", re.IGNORECASE
)
_ID_CAMEL_RE = re.compile(r"[a-z](?:Id|ID)$")  # orderId, customerID (case-sensitive on purpose)
_CODE_RE = re.compile(
    r"(?:^|[_\s\-])(?:zip|zipcode|postal|pincode|pin)(?:$|[_\s\-])", re.IGNORECASE
)
_DATE_LIKE = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)
_INDEX_NAMES = {"index", "#", "sno", "s.no", "sr no", "sr.no", "srno", "serial", "serial no"}


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str
    pandas_dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    stored_as_text: bool = False
    sample_values: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- helpers
def is_text_dtype(series: pd.Series) -> bool:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return False
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _has_id_hint(name: str) -> bool:
    return bool(_ID_WORD_RE.search(name) or _ID_CAMEL_RE.search(name))


def _looks_like_index(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered.startswith("unnamed") or lowered in _INDEX_NAMES


def parse_numeric_text(series: pd.Series) -> pd.Series:
    """Convert text such as '$1,200.50', '(300)', '12%' to floats. Unreadable values -> NaN."""
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    mask = series.notna()
    if not mask.any():
        return out
    t = series[mask].astype(str).str.strip()
    negative = t.str.match(r"^\(.*\)$")  # accounting style (300) means -300
    t = t.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    t = t.str.replace(_CURRENCY_RE, "", regex=True)
    t = t.str.replace("%", "", regex=False).str.replace(r"\s+", "", regex=True)
    # Treat commas as thousands separators only when the pattern is unambiguous
    thousands = t.str.match(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
    t = t.where(~thousands, t.str.replace(",", "", regex=False))
    values = pd.to_numeric(t, errors="coerce").astype("float64")
    values = values.where(~negative, -values)
    out.loc[mask] = values
    return out


def parse_datetime_text(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Convert text dates to datetimes. Unreadable values -> NaT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=dayfirst)


# ---------------------------------------------------------------- detection
def _detect_numeric(non_null: pd.Series, name: str, nunique: int, unique_pct: float) -> str:
    values = non_null.astype("float64")
    if nunique == 2 and set(values.unique()) <= {0.0, 1.0}:
        return BOOLEAN
    if _CODE_RE.search(name):
        return CATEGORICAL
    integer_valued = bool((values % 1 == 0).all())
    if integer_valued and (
        (_has_id_hint(name) and unique_pct >= 0.9)
        or (_looks_like_index(name) and unique_pct >= 0.99)
    ):
        return ID
    return NUMERIC


def _detect_text(
    non_null: pd.Series, name: str, n: int, nunique: int, unique_pct: float
) -> tuple[str, bool]:
    strings = non_null.astype(str).str.strip()

    distinct = set(strings.str.lower().unique())
    if 1 < len(distinct) <= 2 and distinct <= set(BOOL_MAP):
        return BOOLEAN, True
    if _CODE_RE.search(name):
        return CATEGORICAL, False
    if _has_id_hint(name) and unique_pct >= 0.9:
        return ID, False

    sample = strings if n <= 5000 else strings.sample(5000, random_state=0)

    numeric_ok = parse_numeric_text(sample).notna().mean() >= 0.9
    leading_zero = sample.str.match(r"^0\d+$").mean()  # codes like 02134 are not numbers
    if numeric_ok and leading_zero < 0.05:
        return NUMERIC, True

    if sample.str.contains(_DATE_LIKE).mean() >= 0.8:
        if parse_datetime_text(sample).notna().mean() >= 0.9:
            return DATETIME, True

    avg_len = strings.str.len().mean()
    avg_words = strings.str.split().str.len().mean()
    is_long = avg_len > 30 or avg_words > 3
    if is_long and unique_pct >= 0.5:
        return TEXT, False
    if unique_pct >= 0.9 and n >= 20:
        return ID, False
    if nunique <= 20 or unique_pct < 0.5:
        return CATEGORICAL, False
    return TEXT, False


def detect_column_type(series: pd.Series, name: str) -> tuple[str, bool]:
    """Return (inferred_type, stored_as_text) for one column."""
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return EMPTY, False
    nunique = int(non_null.nunique())
    unique_pct = nunique / n
    name_s = str(name)

    if nunique == 1 and n > 1:
        return CONSTANT, False
    if pd.api.types.is_bool_dtype(series):
        return BOOLEAN, False
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME, False
    if pd.api.types.is_numeric_dtype(series):
        return _detect_numeric(non_null, name_s, nunique, unique_pct), False
    if isinstance(series.dtype, pd.CategoricalDtype):
        return CATEGORICAL, False
    if is_text_dtype(series):
        return _detect_text(non_null, name_s, n, nunique, unique_pct)
    return TEXT, False


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    total = len(df)
    profiles: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        inferred, as_text = detect_column_type(s, col)
        missing = int(s.isna().sum())
        unique = int(s.nunique())
        profiles.append(
            ColumnProfile(
                name=str(col),
                inferred_type=inferred,
                pandas_dtype=str(s.dtype),
                missing_count=missing,
                missing_pct=round(missing / total * 100, 2) if total else 0.0,
                unique_count=unique,
                unique_pct=round(unique / total * 100, 2) if total else 0.0,
                stored_as_text=as_text,
                sample_values=[str(v) for v in s.dropna().unique()[:3]],
            )
        )
    return profiles


def columns_of_type(profiles: list[ColumnProfile], *types: str) -> list[str]:
    return [p.name for p in profiles if p.inferred_type in types]


def profiles_to_dataframe(profiles: list[ColumnProfile]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": p.name,
                "detected type": p.inferred_type,
                "stored as text": "yes" if p.stored_as_text else "",
                "pandas dtype": p.pandas_dtype,
                "missing": p.missing_count,
                "missing %": p.missing_pct,
                "unique": p.unique_count,
                "examples": ", ".join(p.sample_values),
            }
            for p in profiles
        ]
    )