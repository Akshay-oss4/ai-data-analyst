"""Tests for the AI layer. No real model is needed: the model is replaced by a fake."""
import json

import numpy as np
import pandas as pd

import src.ai.llm as llm_module
from src.ai.errors import LLMError
from src.ai.guardrails import extract_numbers, find_unsupported_numbers
from src.ai.insights import generate_insights, normalize_llm_output, parse_llm_json
from src.analysis.pipeline import run_analysis
from src.cleaning.cleaner import clean_dataframe


def make_bundle():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "revenue": 100 + 0.25 * np.arange(n) + rng.normal(0, 3, n),
            "units": rng.integers(1, 20, n),
            "region": rng.choice(["North", "South", "East"], n),
        }
    )
    df.loc[[20, 120, 220], "revenue"] = 1000.0
    cleaning = clean_dataframe(df)
    return run_analysis(cleaning), cleaning


FAKE_ANSWER = json.dumps(
    {
        "summary": "Revenue is improving. Sales jumped 999% last month.",
        "insights": [{"title": "Growth", "explanation": "Revenue keeps rising.", "based_on": ["F1", "F99"]}],
        "recommendations": [
            {"action": "Keep investing in what works.", "reason": "Growth is steady.", "priority": "urgent", "based_on": ["F1"]}
        ],
        "caveats": ["Small sample."],
    }
)


def test_extract_numbers_handles_commas_percent_and_scale():
    numbers = extract_numbers("Revenue 1,234.5 grew 12% to 1.23M")
    assert 1234.5 in numbers
    assert 12.0 in numbers
    assert any(abs(v - 1_230_000) < 1 for v in numbers)


def test_find_unsupported_numbers_flags_invented_values():
    facts = "revenue rose 24% to 5.1M over 11 months"
    ai = "Revenue rose 24% to about 5.1 million, and costs fell 37%. There are 3 insights."
    assert find_unsupported_numbers(ai, facts) == ["37%"]


def test_parse_llm_json_handles_fences_and_extra_text():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('Here you go: {"a": 1} thanks') == {"a": 1}
    assert parse_llm_json("not json at all") is None


def test_normalize_drops_unknown_ids_and_bad_priority():
    data = json.loads(FAKE_ANSWER)
    parsed = normalize_llm_output(data, {"F1", "F2"})
    assert parsed["insights"][0].based_on == ["F1"]
    assert parsed["recommendations"][0].priority == "medium"


def test_rule_based_insights_work_without_ai():
    bundle, cleaning = make_bundle()
    result = generate_insights(bundle, cleaning, "x.csv", use_llm=False)
    valid = {f"F{i}" for i in range(1, 13)}
    assert result.source == "rules"
    assert result.summary
    assert result.recommendations
    assert all(set(i.based_on) <= valid for i in result.insights)


def test_llm_path_with_fake_model(monkeypatch):
    bundle, cleaning = make_bundle()
    monkeypatch.setattr(llm_module, "generate", lambda system, prompt, model=None: (FAKE_ANSWER, "fake-model"))
    result = generate_insights(bundle, cleaning, "x.csv")
    assert result.source == "llm"
    assert result.model == "fake-model"
    assert "999%" in result.unsupported_numbers
    assert result.insights[0].based_on == ["F1"]
    assert result.recommendations[0].priority == "medium"


def test_llm_error_falls_back_to_rules(monkeypatch):
    bundle, cleaning = make_bundle()

    def boom(system, prompt, model=None):
        raise LLMError("boom")

    monkeypatch.setattr(llm_module, "generate", boom)
    result = generate_insights(bundle, cleaning, "x.csv")
    assert result.source == "rules"
    assert "boom" in result.note


def test_unreadable_answer_falls_back_to_rules(monkeypatch):
    bundle, cleaning = make_bundle()
    monkeypatch.setattr(llm_module, "generate", lambda system, prompt, model=None: ("not json", "fake-model"))
    result = generate_insights(bundle, cleaning, "x.csv")
    assert result.source == "rules"
    assert "could not be read" in result.note