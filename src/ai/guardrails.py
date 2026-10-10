"""Anti-hallucination check: find numbers in the AI's text that are not in the fact sheet."""
from __future__ import annotations

import re
from collections.abc import Iterator

_DATE_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%|k\b|m\b|b\b|million\b|billion\b|thousand\b)?",
    re.IGNORECASE,
)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


def _iter_numbers(text: str) -> Iterator[tuple[float, str, str]]:
    """Yield (value, suffix, displayed_text) for every number in the text."""
    text = _DATE_RE.sub(" ", text)  # dates are not 'claims'
    for match in _NUMBER_RE.finditer(text):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        yield value, (match.group(2) or "").lower(), match.group(0).strip()


def extract_numbers(text: str) -> set[float]:
    """All numbers in a text, including the scaled value of '1.2M' / '3 million'."""
    known: set[float] = set()
    for value, suffix, _ in _iter_numbers(text):
        known.add(value)
        if suffix in _SCALE:
            known.add(value * _SCALE[suffix])
    return known


def _tolerance(reference: float) -> float:
    magnitude = abs(reference)
    if magnitude < 10:
        return 0.05
    if magnitude < 1000:
        return 0.5
    return 0.05 * magnitude  # rounding like 1.23M -> 1.2M


def _is_trivial(value: float, suffix: str, display: str) -> bool:
    if suffix:
        return False
    if "." not in display and "," not in display and value <= 10:
        return True  # "3 insights", "2 columns"
    return 1900 <= value <= 2100  # years


def find_unsupported_numbers(ai_text: str, fact_text: str) -> list[str]:
    """Numbers the AI wrote that cannot be matched to any number in the fact sheet."""
    known = extract_numbers(fact_text)
    unsupported: list[str] = []
    for value, suffix, display in _iter_numbers(ai_text):
        if _is_trivial(value, suffix, display):
            continue
        candidates = [value]
        if suffix in _SCALE:
            candidates.append(value * _SCALE[suffix])
        if any(abs(abs(c) - abs(k)) <= _tolerance(k) for c in candidates for k in known):
            continue
        if display not in unsupported:
            unsupported.append(display)
    return unsupported