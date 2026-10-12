"""Load CSV / Excel files into pandas DataFrames.

Handles encoding detection, delimiter detection, Excel sheets, and size checks.
All errors that a user can fix are raised as FileLoadError with a friendly message.
"""
from __future__ import annotations
import codecs
import io
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


import pandas as pd

from src.config import settings

CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm"}
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS

# Tried in order. latin-1 accepts any byte, so decoding always succeeds eventually.
ENCODINGS_TO_TRY = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
DELIMITER_CANDIDATES = (",", ";", "\t", "|")


class FileLoadError(Exception):
    """Raised when a file cannot be loaded. The message is safe to show to the user."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    filename: str
    file_type: str  # "csv" or "excel"
    size_bytes: int
    encoding: str | None = None
    delimiter: str | None = None
    sheet_name: str | None = None
    sheet_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def check_file(filename: str, size_bytes: int) -> str:
    """Validate the file name and size. Returns the lower-case extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".xls":
        raise FileLoadError(
            "Old .xls files are not supported. Open the file in Excel and save it as .xlsx."
        )
    if ext not in SUPPORTED_EXTENSIONS:
        raise FileLoadError(
            f"Unsupported file type '{ext or 'none'}'. "
            f"Please upload one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if size_bytes == 0:
        raise FileLoadError("The file is empty (0 bytes).")
    limit = settings.max_upload_mb * 1024 * 1024
    if size_bytes > limit:
        raise FileLoadError(
            f"The file is {size_bytes / 1024 / 1024:.1f} MB, above the "
            f"{settings.max_upload_mb} MB limit."
        )
    return ext


def _decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode bytes to text. Handles UTF-32/UTF-16 byte-order marks and rejects binary files."""
    try:
        if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
            return data.decode("utf-32"), "utf-32"
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return data.decode("utf-16"), "utf-16"
    except UnicodeDecodeError as exc:
        raise FileLoadError("The file has a UTF-16/UTF-32 marker but could not be decoded.") from exc
    if b"\x00" in data[:4096]:
        raise FileLoadError("This does not look like a text file (it contains binary data). Is it really a CSV?")
    for encoding in ENCODINGS_TO_TRY:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise FileLoadError("Could not determine the text encoding of the file.")


def _detect_delimiter(text: str) -> str:
    """Pick the delimiter that appears a consistent number of times on each line."""
    lines = [line for line in text.splitlines()[:20] if line.strip()]
    if not lines:
        return ","
    best, best_score = ",", (0.0, 0)
    for candidate in DELIMITER_CANDIDATES:
        counts = [line.count(candidate) for line in lines]
        common_count, frequency = Counter(counts).most_common(1)[0]
        if common_count == 0:
            continue
        score = (frequency / len(lines), common_count)
        if score > best_score:
            best, best_score = candidate, score
    return best


def _read_csv(data: bytes, ext: str) -> tuple[pd.DataFrame, str, str, list[str]]:
    text, encoding = _decode_bytes(data)
    delimiter = "\t" if ext == ".tsv" else _detect_delimiter(text)
    notes: list[str] = []
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError as exc:
        raise FileLoadError("The file contains no readable data.") from exc
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(
                io.StringIO(text), sep=delimiter, engine="python", on_bad_lines="skip"
            )
            notes.append("Some malformed rows were skipped while reading the file.")
        except Exception as exc:  # noqa: BLE001
            raise FileLoadError(f"Could not parse the file as CSV: {exc}") from exc
    return df, encoding, delimiter, notes


def get_excel_sheet_names(data: bytes) -> list[str]:
    """Return the sheet names of an Excel workbook."""
    try:
        with pd.ExcelFile(io.BytesIO(data), engine="openpyxl") as workbook:
            return list(workbook.sheet_names)
    except Exception as exc:  # noqa: BLE001
        raise FileLoadError(f"Could not open the Excel file: {exc}") from exc


def load_file(data: bytes, filename: str, sheet_name: str | None = None) -> LoadResult:
    """Load an uploaded file (as raw bytes) into a LoadResult."""
    ext = check_file(filename, len(data))

    if ext in EXCEL_EXTENSIONS:
        names = get_excel_sheet_names(data)
        if not names:
            raise FileLoadError("The Excel file contains no sheets.")
        chosen = sheet_name if sheet_name in names else names[0]
        try:
            df = pd.read_excel(io.BytesIO(data), sheet_name=chosen, engine="openpyxl")
        except Exception as exc:  # noqa: BLE001
            raise FileLoadError(f"Could not read sheet '{chosen}': {exc}") from exc
        return LoadResult(
            df=df,
            filename=filename,
            file_type="excel",
            size_bytes=len(data),
            sheet_name=chosen,
            sheet_names=names,
        )

    df, encoding, delimiter, notes = _read_csv(data, ext)
    return LoadResult(
        df=df,
        filename=filename,
        file_type="csv",
        size_bytes=len(data),
        encoding=encoding,
        delimiter=delimiter,
        notes=notes,
    )