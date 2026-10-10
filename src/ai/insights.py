"""Generate business insights and recommendations from the computed facts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.ai import llm
from src.ai.context import build_fact_sheet
from src.ai.errors import LLMError
from src.ai.guardrails import find_unsupported_numbers
from src.ai.prompts import SYSTEM_PROMPT, build_user_prompt
from src.analysis.findings import Finding
from src.analysis.pipeline import AnalysisBundle
from src.cleaning.cleaner import CleaningResult

PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Insight:
    title: str
    explanation: str
    based_on: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    action: str
    reason: str
    priority: str
    based_on: list[str] = field(default_factory=list)


@dataclass
class InsightsResult:
    source: str  # "llm" or "rules"
    model: str
    summary: str
    insights: list[Insight]
    recommendations: list[Recommendation]
    caveats: list[str]
    unsupported_numbers: list[str] = field(default_factory=list)
    note: str = ""
    fact_sheet: str = ""


# ------------------------------------------------------------------ parsing
def _outer_braces(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else ""


def parse_llm_json(text: str) -> dict | None:
    """Parse the model's answer, tolerating code fences and extra words around the JSON."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    for candidate in (cleaned, _outer_braces(cleaned)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _items(data: dict, key: str) -> list:
    value = data.get(key)
    return value if isinstance(value, list) else []


def _valid_ids(value, valid_ids: set[str]) -> list[str]:
    ids = value if isinstance(value, list) else []
    return [i for i in ids if isinstance(i, str) and i in valid_ids]


def normalize_llm_output(data: dict, valid_ids: set[str]) -> dict | None:
    """Clean up the model's JSON. Returns None if it is unusable."""
    summary = _text(data.get("summary"))

    insights: list[Insight] = []
    for item in _items(data, "insights"):
        if not isinstance(item, dict):
            continue
        title, explanation = _text(item.get("title")), _text(item.get("explanation"))
        if title and explanation:
            insights.append(Insight(title, explanation, _valid_ids(item.get("based_on"), valid_ids)))

    recommendations: list[Recommendation] = []
    for item in _items(data, "recommendations"):
        if not isinstance(item, dict):
            continue
        action = _text(item.get("action"))
        if not action:
            continue
        priority = _text(item.get("priority")).lower()
        recommendations.append(
            Recommendation(
                action,
                _text(item.get("reason")),
                priority if priority in PRIORITY_RANK else "medium",
                _valid_ids(item.get("based_on"), valid_ids),
            )
        )
    recommendations.sort(key=lambda r: PRIORITY_RANK[r.priority])

    caveats = [c.strip() for c in _items(data, "caveats") if isinstance(c, str) and c.strip()]

    if not summary or not (insights or recommendations):
        return None
    return {
        "summary": summary,
        "insights": insights,
        "recommendations": recommendations,
        "caveats": caveats,
    }


# ------------------------------------------------------------------ LLM path
def _from_llm(fact_sheet: str, valid_ids: set[str], model: str | None) -> tuple[dict, str]:
    prompt = build_user_prompt(fact_sheet)
    text, label = llm.generate(SYSTEM_PROMPT, prompt, model)
    data = parse_llm_json(text)
    parsed = normalize_llm_output(data, valid_ids) if data else None

    if parsed is None:  # one retry with a stricter reminder
        retry = prompt + (
            "\n\nYour previous answer was not valid or complete JSON. "
            "Reply again with ONLY the JSON object in the exact structure above."
        )
        text, label = llm.generate(SYSTEM_PROMPT, retry, model)
        data = parse_llm_json(text)
        parsed = normalize_llm_output(data, valid_ids) if data else None

    if parsed is None:
        raise LLMError("The model's answer could not be read. Try again or choose a larger model.")
    return parsed, label


# ------------------------------------------------------------------ rule-based fallback
def _recommendation(fid: str, f: Finding) -> Recommendation:
    severity = {"high": "high", "medium": "medium"}.get(f.severity, "low")
    col = f.column or "this metric"

    if f.category == "trend":
        if f.title.endswith("declining"):
            return Recommendation(
                f"Find out what changed in {col}: compare regions, products or customer groups to see where the decline comes from.",
                "It has been falling along its trend line.", severity, [fid])
        return Recommendation(
            f"Identify what is driving the improvement in {col} and keep doing it.",
            "It has been rising along its trend line.", "low", [fid])

    if f.category == "anomaly":
        if f.title.startswith("Unusual"):
            period = f.evidence.get("period", "that period")
            return Recommendation(
                f"Check what happened in {period} (campaign, stock issue, price change or a data error).",
                "That period is far from where its trend line says it should be.", severity, [fid])
        if f.column:
            return Recommendation(
                f"Review the unusual values in {col}: correct them if they are data-entry errors, or document them if they are real events.",
                "Some values are far outside the normal range.", severity, [fid])
        return Recommendation(
            "Look at the flagged unusual rows together to see whether they share a cause such as one customer, region or product.",
            "These rows look unusual when several columns are combined.", severity, [fid])

    if f.category == "correlation":
        return Recommendation(
            "Test this relationship with a small experiment before acting on it.",
            "Two columns move together, but that alone does not prove one causes the other.", "low", [fid])

    return Recommendation(
        "Fix this data problem at the source so future analyses start from cleaner data.",
        f"{f.title}.", "medium" if severity == "high" else severity, [fid])


def rule_based_insights(
    cleaning: CleaningResult, numbered: list[tuple[str, Finding]]
) -> InsightsResult:
    findings = [f for _, f in numbered]
    rows, cols = cleaning.after_shape
    n_high = sum(f.severity == "high" for f in findings)
    n_medium = sum(f.severity == "medium" for f in findings)

    summary = (
        f"After cleaning, the dataset has {rows:,} rows and {cols} columns. "
        f"The analysis produced {len(findings)} findings: {n_high} high priority and {n_medium} medium priority."
    )
    if findings:
        summary += f" The most important one: {findings[0].title}."

    insights = [Insight(f.title, f.detail, [fid]) for fid, f in numbered[:5]]

    recommendations: list[Recommendation] = []
    seen: set[tuple] = set()
    for fid, f in numbered:
        key = (f.category, f.column or f.title)
        if key in seen:
            continue
        seen.add(key)
        recommendations.append(_recommendation(fid, f))
        if len(recommendations) == 5:
            break
    if not recommendations:
        recommendations.append(
            Recommendation(
                "Re-run the analysis when more data (or a longer time period) is available.",
                "Nothing notable stood out in this dataset.", "low", [])
        )
    recommendations.sort(key=lambda r: PRIORITY_RANK[r.priority])

    return InsightsResult(
        source="rules",
        model="built-in rules",
        summary=summary,
        insights=insights,
        recommendations=recommendations,
        caveats=["These are statistical signals, not proof of cause. A person should judge each one."],
    )


# ------------------------------------------------------------------ public API
def generate_insights(
    bundle: AnalysisBundle,
    cleaning: CleaningResult,
    filename: str,
    model: str | None = None,
    use_llm: bool = True,
) -> InsightsResult:
    fact_sheet, numbered = build_fact_sheet(bundle, cleaning, filename)
    valid_ids = {fid for fid, _ in numbered}
    note = ""

    if use_llm:
        try:
            parsed, label = _from_llm(fact_sheet, valid_ids, model)
        except LLMError as exc:
            note = str(exc)
        else:
            all_text = " ".join(
                [parsed["summary"]]
                + [f"{i.title} {i.explanation}" for i in parsed["insights"]]
                + [f"{r.action} {r.reason}" for r in parsed["recommendations"]]
                + parsed["caveats"]
            )
            return InsightsResult(
                source="llm",
                model=label,
                summary=parsed["summary"],
                insights=parsed["insights"],
                recommendations=parsed["recommendations"],
                caveats=parsed["caveats"],
                unsupported_numbers=find_unsupported_numbers(all_text, fact_sheet),
                fact_sheet=fact_sheet,
            )

    result = rule_based_insights(cleaning, numbered)
    result.note = note or "AI was switched off."
    result.fact_sheet = fact_sheet
    return result