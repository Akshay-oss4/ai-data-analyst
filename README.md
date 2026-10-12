# AI-Powered Automated Data Analyst

Upload a CSV or Excel file and get an automatic analysis: validation, cleaning, KPIs, trends, anomaly detection,
AI-written business insights, an interactive dashboard, a downloadable report, and a history of past analyses.
Works on any tabular dataset. Nothing is hard-coded to one file. Free and open source, and the AI runs
locally with Ollama, so your data stays on your computer.

![Overview](docs/screenshots/overview.png)

## Features

- **Smart loading:** CSV, TSV, TXT and Excel (several sheets); detects encoding (UTF-8, Windows-1252, UTF-16) and delimiter
- **Automatic column types:** numeric, categorical, datetime, boolean, ID, text, including numbers and dates stored as text (`"$1,200"`, `"12%"`, `"2024-03-01"`)
- **Cleaning with a full log:** trims text, standardizes N/A values, converts types, removes duplicates, fills missing values
- **KPIs and EDA:** metrics chosen automatically, distributions, correlations, categories, charts
- **Trends:** improving, declining or stable, with a significance test, and incomplete periods excluded
- **Anomalies:** IQR, Z-score, Isolation Forest (with a NumPy fallback), and unusual periods
- **AI insights:** the model only sees a list of computed facts, and every number it writes is checked against them
- **Dashboard:** six pages, filters (date range, categories), KPI cards with trend arrows
- **Reports:** a self-contained HTML report (print it to PDF) and an Excel workbook
- **History:** SQLite storage, compare two analyses, chart a KPI over time

| Analysis | Insights |
|---|---|
| ![Analysis](docs/screenshots/analysis.png) | ![Insights](docs/screenshots/insights.png) |
| **Report** | **History** |
| ![Report](docs/screenshots/report.png) | ![History](docs/screenshots/history.png) |

## How it works

```
Upload -> Validation -> Type detection -> Cleaning -> KPIs / EDA / Trends / Anomalies
      -> Structured findings -> AI explanation (local LLM, with number checking)
      -> Dashboard, HTML/Excel report, SQLite history
```

Design decisions:

- **The AI explains, it doesn't calculate.** The analysis engine computes every number. The language model only
  receives them as a fact sheet, must cite finding IDs, and its output is checked for numbers that aren't in the facts.
- **It always works without AI.** If Ollama is off or its answer is unreadable, rule-based insights are used.
- **Heuristics are visible.** Detected types, cleaning steps and thresholds are shown, so you can check what happened.

## Quick start (Windows)

```powershell
git clone https://github.com/Akshay-oss4/ai-data-analyst.git
cd ai-data-analyst
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

For AI-written insights install [Ollama](https://ollama.com) and run `ollama pull llama3.2`.
Sample files: `python scripts\generate_sample_data.py` and `python scripts\generate_messy_data.py` create files in `data/sample`.

## Tests

```powershell
python -m pytest
```

The suite covers each module, every sample file end to end, and about 20 awkward files (one row, UTF-16, infinite values,
huge numbers, control characters, wide tables). A GitHub Actions workflow runs it on every push.
`python scripts\benchmark.py 100000` times each stage on a large synthetic file.

## Project structure

```
app.py                  Streamlit entry point (navigation, upload, filters)
src/ingestion/          loading and validation
src/profiling/          column type detection
src/cleaning/           cleaning pipeline with log
src/analysis/           KPIs, EDA, trends, anomalies, findings, charts, pipeline runner
src/ai/                 Ollama client, prompts, fact sheet, number checker, insights
src/reporting/          HTML and Excel reports
src/database/           SQLite history
src/ui/                 dashboard pages and components
tests/                  unit, end-to-end and edge-case tests
scripts/                sample data generators and benchmark
```

## Limitations

- Column types and metric roles are heuristics (names such as `revenue` or `cost` guide them) and can be wrong.
- Dates like `03/04/2024` are read month-first; European decimals like `1.234,56` are not parsed.
- Local language models are less capable than large hosted ones; their text should be reviewed.
- Anomalies are statistical signals, not proof of errors or cause.
- History is stored in a local SQLite file (one user). PostgreSQL would be the next step for sharing.

## Roadmap

PostgreSQL support, a manual override for column types, seasonality-aware trends, natural-language questions about the data,
and Docker packaging.