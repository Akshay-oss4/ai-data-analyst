"""Tests for file loading and validation."""
import io

import pandas as pd
import pytest

from src.ingestion.loader import FileLoadError, load_file
from src.ingestion.validator import validate_dataframe


def test_load_comma_csv():
    result = load_file(b"a,b,c\n1,2,3\n4,5,6\n", "x.csv")
    assert result.df.shape == (2, 3)
    assert result.delimiter == ","


def test_load_semicolon_csv():
    result = load_file(b"a;b;c\n1;2;3\n4;5;6\n", "x.csv")
    assert result.delimiter == ";"
    assert list(result.df.columns) == ["a", "b", "c"]


def test_load_cp1252_encoding():
    data = "name;city\nJosé;Málaga\n".encode("cp1252")
    result = load_file(data, "x.csv")
    assert result.encoding == "cp1252"
    assert "José" in result.df["name"].values


def test_load_excel_with_sheets():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="One", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(writer, sheet_name="Two", index=False)
    result = load_file(buffer.getvalue(), "x.xlsx", sheet_name="Two")
    assert result.sheet_names == ["One", "Two"]
    assert result.df.shape == (3, 1)


def test_empty_file_rejected():
    with pytest.raises(FileLoadError):
        load_file(b"", "x.csv")


def test_unsupported_extension_rejected():
    with pytest.raises(FileLoadError):
        load_file(b"hello", "x.pdf")


def test_header_only_file_is_invalid():
    result = load_file(b"a,b\n", "x.csv")
    report = validate_dataframe(result.df)
    assert not report.is_valid


def test_validator_reports_duplicates():
    df = pd.DataFrame({"a": range(12), "b": range(12)})
    df = pd.concat([df, df.head(3)], ignore_index=True)
    report = validate_dataframe(df)
    assert report.is_valid
    assert report.info["duplicate_rows"] == 3