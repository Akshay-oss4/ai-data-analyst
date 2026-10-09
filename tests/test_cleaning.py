"""Tests for the cleaning pipeline."""
import numpy as np
import pandas as pd

from src.cleaning.cleaner import CleaningOptions, clean_dataframe


def _messy_df() -> pd.DataFrame:
    n = 20
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(n)],
            " amount ": [f"${100 + i * 10:,.2f}" for i in range(n)],
            "joined": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
            "city": ["Pune", " Pune ", "Delhi", "N/A"] * 5,
            "empty": [np.nan] * n,
        }
    )


def test_pipeline_converts_and_cleans():
    out = clean_dataframe(_messy_df()).df
    assert "amount" in out.columns  # header trimmed
    assert "empty" not in out.columns  # fully empty column dropped
    assert pd.api.types.is_numeric_dtype(out["amount"])
    assert pd.api.types.is_datetime64_any_dtype(out["joined"])
    assert out["city"].isna().sum() == 0
    assert set(out["city"]) == {"Pune", "Delhi", "Unknown"}
    assert len(out) == 20


def test_duplicates_removed():
    df = pd.DataFrame({"a": [1, 2, 2, 3] * 5, "b": ["x", "y", "y", "z"] * 5})
    assert len(clean_dataframe(df).df) == 3


def test_duplicates_kept_when_option_off():
    df = pd.DataFrame({"a": [1, 2, 2, 3] * 5, "b": ["x", "y", "y", "z"] * 5})
    result = clean_dataframe(df, CleaningOptions(drop_duplicates=False))
    assert len(result.df) == 20


SCORES = [1.0, 2.0, 3.0, np.nan, 5.0, 4.0, 2.5, 3.5, 4.5, 1.5, 2.2, 3.3]


def test_numeric_missing_filled_with_median():
    result = clean_dataframe(pd.DataFrame({"score": SCORES}))
    assert result.df["score"].isna().sum() == 0


def test_numeric_missing_left_when_option_none():
    result = clean_dataframe(pd.DataFrame({"score": SCORES}), CleaningOptions(numeric_fill="none"))
    assert result.df["score"].isna().sum() == 1


def test_sparse_column_dropped():
    df = pd.DataFrame({"keep": range(10), "sparse": [1.0] + [np.nan] * 9})
    result = clean_dataframe(df)
    assert "sparse" not in result.df.columns
    assert "keep" in result.df.columns


def test_input_dataframe_not_modified():
    df = _messy_df()
    before = df.copy()
    clean_dataframe(df)
    pd.testing.assert_frame_equal(df, before)