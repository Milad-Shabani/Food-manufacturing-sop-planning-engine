#!/usr/bin/env python3
"""Export the planning warehouse into dashboard/data.js.

Run this after `python -m sop_planning.cli run` any time you want the
static dashboard (dashboard/index.html) to reflect a fresh pipeline run —
e.g. after regenerating the dataset with a different seed.

    python scripts/export_dashboard_data.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

WAREHOUSE = Path("data/processed/planning_warehouse.db")
RAW_DIR = Path("data/raw")
OUT_FILE = Path("dashboard/data.js")


def main() -> None:
    if not WAREHOUSE.exists():
        raise SystemExit(f"{WAREHOUSE} not found — run `python -m sop_planning.cli run` first.")

    conn = sqlite3.connect(WAREHOUSE)
    weekly = pd.read_sql("select * from fact_sop_weekly_summary order by week_start", conn)
    accuracy = pd.read_sql("select * from fact_forecast_accuracy", conn)
    plan = pd.read_sql("select * from fact_production_plan", conn)
    ccc = pd.read_sql("select * from fact_working_capital order by week_start", conn)

    products = pd.read_csv(RAW_DIR / "dim_product.csv")
    lines = pd.read_csv(RAW_DIR / "dim_production_line.csv")

    # Kolucheh line detail (the line the README calls out as the bottleneck;
    # swap the filter below if a different line becomes the story in your run).
    line_week = (
        plan.groupby(["line_id", "week_start"], as_index=False)
        .agg(
            allocated_units=("allocated_units", "sum"),
            forecast_units=("forecast_units", "sum"),
            unmet_units=("unmet_units", "sum"),
        )
        .merge(lines[["line_id", "line_name"]], on="line_id")
    )
    bottleneck_line_id = (
        line_week.loc[line_week["unmet_units"].idxmax(), "line_id"]
        if line_week["unmet_units"].sum()
        else line_week["line_id"].iloc[0]
    )
    l3 = line_week[line_week["line_id"] == bottleneck_line_id].sort_values("week_start")

    plan_products = plan.merge(
        products[["product_id", "product_name", "unit_price_irr", "category"]], on="product_id"
    )
    plan_products["revenue_lost"] = plan_products["unmet_units"] * plan_products["unit_price_irr"]
    top_unmet = (
        plan_products[plan_products["unmet_units"] > 0]
        .groupby(["product_id", "product_name", "category"], as_index=False)
        .agg(unmet_units=("unmet_units", "sum"), revenue_lost=("revenue_lost", "sum"))
        .sort_values("revenue_lost", ascending=False)
        .head(8)
    )

    acc_per_product = (
        accuracy[accuracy["product_id"] != "__OVERALL__"]
        .merge(products[["product_id", "product_name"]], on="product_id")[
            ["product_name", "wape", "mape"]
        ]
        .sort_values("wape")
    )
    overall_wape = float(accuracy.loc[accuracy["product_id"] == "__OVERALL__", "wape"].iloc[0])
    avg_ccc = float(ccc["ccc"].tail(12).mean()) if not ccc.empty else None

    payload = {
        "weekly": weekly.to_dict(orient="records"),
        "l3": l3.to_dict(orient="records"),
        "top_unmet": top_unmet.to_dict(orient="records"),
        "acc": acc_per_product.to_dict(orient="records"),
        "ccc": ccc.to_dict(orient="records"),
        "overall_wape": overall_wape,
        "kpi": {
            "total_forecast": int(weekly["forecast_units"].sum()),
            "total_allocated": int(weekly["allocated_units"].sum()),
            "total_revenue_lost": int(weekly["revenue_lost_irr"].sum()),
            "min_fill_rate": float(weekly["fill_rate"].min()),
            "avg_ccc": avg_ccc,
        },
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        "const DASHBOARD_DATA = " + json.dumps(payload, ensure_ascii=False, default=str) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_FILE} ({len(weekly)} weeks, bottleneck line: {bottleneck_line_id})")


if __name__ == "__main__":
    main()
