# AI-Powered Automated Data Analyst

Upload a CSV or Excel file and get automatic cleaning, EDA, KPIs, trend and
anomaly detection, AI-written business insights, a dashboard, and a downloadable report.

## Status
Under active development (Phase 2: project skeleton).

## Quick start (Windows)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    copy .env.example .env
    streamlit run app.py

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, Ollama (local LLM), SQLite.