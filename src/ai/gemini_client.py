"""Optional client for Google's Gemini API. Only used if AI_PROVIDER=gemini and a key is set.

Free-tier terms, limits and model names change often: check ai.google.dev before relying on this.
Only the aggregated fact sheet is sent, never raw rows.
"""
from __future__ import annotations

import requests

from src.ai.errors import LLMError
from src.config import settings


def generate(system: str, prompt: str, json_mode: bool = True) -> str:
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set in .env.")
    model = settings.gemini_model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    generation_config = {"temperature": 0.2}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    headers = {"x-goog-api-key": settings.gemini_api_key, "Content-Type": "application/json"}

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=60)
    except requests.RequestException as exc:
        raise LLMError("Could not reach the Gemini API.") from exc

    if resp.status_code == 429:
        raise LLMError("Gemini free-tier rate limit reached. Wait a minute, or switch back to Ollama.")
    if resp.status_code in (400, 401, 403):
        raise LLMError(
            "Gemini rejected the request. Check GEMINI_API_KEY, the model name, and availability in your region."
        )
    if resp.status_code == 404:
        raise LLMError(f"Gemini model '{model}' was not found. Check current model names on ai.google.dev.")
    if not resp.ok:
        raise LLMError(f"Gemini returned an error (HTTP {resp.status_code}).")
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError("Gemini returned an unexpected response (it may have blocked the answer).") from exc