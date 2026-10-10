"""Minimal client for a local Ollama server (https://ollama.com)."""
from __future__ import annotations

import requests

from src.ai.errors import LLMError
from src.config import settings


def list_models(host: str | None = None, timeout: float = 3.0) -> list[str] | None:
    """Names of downloaded models, or None if the server cannot be reached."""
    host = host or settings.ollama_host
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except (requests.RequestException, ValueError, KeyError):
        return None


def generate(system: str, prompt: str, model: str | None = None, json_mode: bool = True) -> str:
    """Send one request to the local model and return its text answer."""
    model = model or settings.ollama_model
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,  # low = more factual, less creative
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": 1200,
        },
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(
            f"{settings.ollama_host}/api/chat", json=payload, timeout=settings.ollama_timeout
        )
    except requests.ConnectionError as exc:
        raise LLMError("Cannot reach Ollama. Start the Ollama app and try again.") from exc
    except requests.Timeout as exc:
        raise LLMError(
            f"Ollama did not answer within {settings.ollama_timeout} seconds. "
            "Try a smaller model (for example llama3.2:1b) or raise OLLAMA_TIMEOUT in .env."
        ) from exc

    if resp.status_code == 404:
        raise LLMError(f"Model '{model}' is not downloaded. Run: ollama pull {model}")
    if not resp.ok:
        raise LLMError(f"Ollama returned an error (HTTP {resp.status_code}).")
    try:
        return resp.json()["message"]["content"]
    except (ValueError, KeyError) as exc:
        raise LLMError("Ollama returned an unexpected response.") from exc