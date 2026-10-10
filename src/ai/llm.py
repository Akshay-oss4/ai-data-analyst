"""One entry point for text generation. Chooses Ollama (default) or Gemini (optional)."""
from __future__ import annotations

from src.ai import gemini_client, ollama_client
from src.config import settings


def active_provider() -> str:
    return "gemini" if settings.ai_provider == "gemini" and settings.gemini_api_key else "ollama"


def generate(system: str, prompt: str, model: str | None = None) -> tuple[str, str]:
    """Return (answer_text, label_describing_which_model_answered)."""
    if active_provider() == "gemini":
        return gemini_client.generate(system, prompt), f"Gemini ({settings.gemini_model})"
    used = model or settings.ollama_model
    return ollama_client.generate(system, prompt, used), f"{used} (local, Ollama)"