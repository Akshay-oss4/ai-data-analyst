"""Generate sample datasets for testing. Run from the project root:
    python scripts\\generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- 1. Realistic sales data (comma CSV, with problems on purpose) ----------
n = 1500
dates = pd.date_range("2024-01-01", periods=365, freq="D")
sales = pd.DataFrame(
    {
        "order_id": np.arange(1, n + 1),
        "order_date": rng.choice(dates, n),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "product": rng.choice(["Laptop", "Phone", "Tablet", "Headphones"], n),
        "units": rng.integers(1, 10, n),
    }
)
price = sales["product"].map({"Laptop": 999.0, "Phone": 699.0, "Tablet": 449.0, "Headphones": 129.0})
sales["unit_price"] = price
growth = 1 + (sales["order_date"] - dates[0]).dt.days / 365 * 0.3  # upward trend
sales["revenue"] = (sales["units"] * price * growth + rng.normal(0, 20, n)).round(2)
sales["customer_rating"] = np.round(rng.normal(4.0, 0.7, n).clip(1, 5), 1)

# Problems: outliers, missing values, duplicates
outlier_idx = rng.choice(n, 8, replace=False)
sales.loc[outlier_idx, "revenue"] *= 8
sales.loc[rng.random(n) < 0.08, "customer_rating"] = np.nan
sales.loc[rng.random(n) < 0.03, "region"] = np.nan
sales = pd.concat([sales, sales.sample(20, random_state=1)], ignore_index=True)
sales = sales.sort_values("order_date").reset_index(drop=True)
sales.to_csv(OUT / "sales_data.csv", index=False)

# ---------- 2. Semicolon-delimited, Windows-1252 encoded (accents) ----------
french = pd.DataFrame(
    {
        "Produit": ["Café", "Thé", "Chocolat chaud"] * 10,
        "Ville": ["Paris", "Lyon", "Nice"] * 10,
        "Quantité": rng.integers(1, 20, 30),
    }
)
french.to_csv(OUT / "ventes_semicolon_cp1252.csv", sep=";", index=False, encoding="cp1252")

# ---------- 3. Excel workbook with two sheets ----------
with pd.ExcelWriter(OUT / "sales_multi_sheet.xlsx", engine="openpyxl") as writer:
    sales.head(300).to_excel(writer, sheet_name="Sales", index=False)
    summary = sales.groupby("region", as_index=False)["revenue"].sum()
    summary.to_excel(writer, sheet_name="Summary", index=False)

# ---------- 4. Header only (should produce a validation error) ----------
(OUT / "header_only.csv").write_text("a,b,c\n", encoding="utf-8")

print("Sample files created in:", OUT)
for f in sorted(OUT.iterdir()):
    print(" -", f.name)