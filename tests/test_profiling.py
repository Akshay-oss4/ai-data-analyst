"""Tests for column type detection."""
import numpy as np
import pandas as pd

from src.profiling.type_detector import detect_column_type, parse_numeric_text


def test_parse_numeric_text_handles_currency_commas_percent_and_negatives():
    s = pd.Series(["$1,200.50", "(300)", "12%", "abc", None])
    out = parse_numeric_text(s)
    assert out.iloc[0] == 1200.50
    assert out.iloc[1] == -300
    assert out.iloc[2] == 12
    assert np.isnan(out.iloc[3])
    assert np.isnan(out.iloc[4])


def test_detect_numeric_stored_as_text():
    s = pd.Series(["$1,000", "$2,500", "$3,100", "$900"] * 5)
    assert detect_column_type(s, "price") == ("numeric", True)


def test_detect_datetime_stored_as_text():
    s = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"] * 5)
    assert detect_column_type(s, "when") == ("datetime", True)


def test_detect_boolean_stored_as_text():
    s = pd.Series(["Yes", "No", "Yes", "No"] * 5)
    assert detect_column_type(s, "flag") == ("boolean", True)


def test_detect_id_column():
    s = pd.Series(range(1, 101))
    assert detect_column_type(s, "order_id")[0] == "id"


def test_detect_categorical_column():
    s = pd.Series(["a", "b", "c"] * 20)
    assert detect_column_type(s, "grade")[0] == "categorical"


def test_detect_constant_and_empty_columns():
    assert detect_column_type(pd.Series([5, 5, 5, 5]), "x")[0] == "constant"
    assert detect_column_type(pd.Series([np.nan] * 3), "y")[0] == "empty"