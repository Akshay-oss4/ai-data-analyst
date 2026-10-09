"""Generate a deliberately messy dataset to test cleaning. Run from anywhere:
    python scripts\\generate_messy_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
n = 200

messy = pd.DataFrame(
    {
        "Customer ID": [f"C{1000 + i}" for i in range(n)],
        " Signup Date ": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Order Value": [f"${v:,.2f}" for v in rng.gamma(3, 120, n)],
        "Discount %": [f"{v}%" for v in rng.integers(0, 30, n)],
        "Is Member": rng.choice(["Yes", "No"], n),
        "City": rng.choice(["Mumbai", "Delhi", "Pune", "Nagpur", "  Mumbai ", "Delhi"], n),
        "Age": rng.integers(18, 70, n).astype(float),
        "Empty Column": np.nan,
    }
)

# Missing values in different disguises
messy.loc[rng.random(n) < 0.10, "Age"] = np.nan
messy.loc[rng.random(n) < 0.05, "Order Value"] = "N/A"
messy.loc[rng.random(n) < 0.06, "City"] = ""
messy.loc[rng.random(n) < 0.04, "Is Member"] = "NA"

# Duplicate rows
messy = pd.concat([messy, messy.sample(10, random_state=3)], ignore_index=True)

messy.to_csv(OUT / "messy_data.csv", index=False)
print("Created:", OUT / "messy_data.csv", "-", len(messy), "rows")