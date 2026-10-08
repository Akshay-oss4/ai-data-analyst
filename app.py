"""Streamlit entry point. For now: a health-check page that proves the setup works."""
import sys

import pandas as pd
import requests
import streamlit as st

from src.config import settings

st.set_page_config(page_title="AI-Powered Automated Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Automated Data Analyst")
st.caption("Phase 2: setup check. The real application is built in the next phases.")

st.subheader("System health check")


def check_ollama() -> tuple[bool, str]:
    """Return (is_running, message) for the local Ollama server."""
    try:
        resp = requests.get(f"{settings.ollama_host}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not models:
            return True, "Ollama is running, but no models are downloaded yet."
        return True, "Ollama is running. Models: " + ", ".join(models)
    except requests.RequestException as exc:
        return False, f"Cannot reach Ollama at {settings.ollama_host} ({exc.__class__.__name__})"


col1, col2, col3 = st.columns(3)
col1.metric("Python", sys.version.split()[0])
col2.metric("Pandas", pd.__version__)
col3.metric("Configured AI model", settings.ollama_model)

ok, message = check_ollama()
(st.success if ok else st.error)(message)

st.divider()
st.write("If you can see this page and the Ollama status above, your environment is ready.")