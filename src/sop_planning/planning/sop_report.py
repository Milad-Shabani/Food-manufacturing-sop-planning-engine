"""Consolidated weekly S&OP summary: the single table a planning meeting
would actually look at, combining demand, production, revenue impact, and
material risk into one row per week.
"""

from __future__ import annotations

import pandas as pd


def build_weekly_summary(
    production_plan: pd.DataFrame, products: pd.DataFrame, mrp: pd.DataFrame
) -> pd.DataFrame:
    prices = products.set_index("product_id")["unit_price_irr"]

    plan = production_plan.copy()
    plan["revenue_captured_irr"] = plan["allocated_units"] * plan["product_id"].map(prices)
    plan["revenue_lost_irr"] = plan["unmet_units"] * plan["product_id"].map(prices)

    weekly = (
        plan.groupby("week_start")
        .agg(
            forecast_units=("forecast_units", "sum"),
            allocated_units=("allocated_units", "sum"),
            unmet_units=("unmet_units", "sum"),
            revenue_captured_irr=("revenue_captured_irr", "sum"),
            revenue_lost_irr=("revenue_lost_irr", "sum"),
        )
        .reset_index()
    )
    weekly["fill_rate"] = (weekly["allocated_units"] / weekly["forecast_units"]).round(4)

    if not mrp.empty:
        shortage_weeks = (
            mrp[mrp["below_safety_stock"]]
            .groupby("week_start")["material_id"]
            .nunique()
            .rename("materials_below_safety_stock")
        )
        weekly = weekly.merge(shortage_weeks, on="week_start", how="left")
        weekly["materials_below_safety_stock"] = (
            weekly["materials_below_safety_stock"].fillna(0).astype(int)
        )
    else:
        weekly["materials_below_safety_stock"] = 0

    return weekly.sort_values("week_start").reset_index(drop=True)
