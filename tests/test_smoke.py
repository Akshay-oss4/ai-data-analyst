"""Smoke tests: confirm the project imports and config loads."""
from src.config import settings


def test_settings_load():
    assert settings.ollama_model
    assert settings.max_upload_mb > 0


def test_core_libraries_import():
    import numpy, pandas, plotly, sklearn, streamlit  # noqa: F401