"""Waste (scrap), plant economics and the value-opportunity ledger, in USD.

* `waste_detail`: scrap by line / product / month, costed at material + direct
  labor per unit, and what bringing every product down to a target scrap rate
  would save per year.
* `plant_economics`: last-full-year P&L view (revenue, materials, payroll),
  line profitability and contribution per machine hour.
* `value_ledger`: one table that puts the cash and annual savings found by
  the engine side by side (inventory policy, scrap, bottleneck, cash cycle).

Scenario targets are documented assumptions (`SCRAP_TARGET`, `BOTTLENECK_GAIN`).
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from sop_planning.config import load_settings

SCRAP_TARGET = 0.02  # every product above 2% scrap is brought down to 2%
BOTTLENECK_GAIN = 0.30  # +30% effective capacity on the binding line (extra shift hours)
WORK_HOURS_PER_MONTH = 21.7 * 8


def _unit_costs(raw: str) -> pd.DataFrame:
    cfg = load_settings()
    lines = pd.read_csv(f"{raw}/dim_production_line.csv")
    prod = pd.read_csv(f"{raw}/dim_product.csv")
    bom = pd.read_csv(f"{raw}/bom.csv")
    mat = pd.read_csv(f"{raw}/dim_raw_material.csv")[["material_id", "unit_cost_usd"]]
    emp = pd.read_csv(f"{raw}/dim_employee.csv")
    m = bom.merge(mat, on="material_id").assign(c=lambda d: d.qty_per_unit * d.unit_cost_usd)
    prod["material_cost"] = prod.product_id.map(m.groupby("product_id").c.sum())
    op = (
        emp[emp.role == "Production Operator"].groupby("production_line_id").monthly_wage_usd.mean()
    )
    ln = lines.set_index("line_id")
    hourly = op / WORK_HOURS_PER_MONTH
    labor = {
        lid: cfg.operators_per_running_hour[lid] * hourly[lid] / ln.capacity_units_per_hour[lid]
        for lid in ln.index
    }
    prod["labor_cost"] = prod.primary_line_id.map(labor)
    prod["unit_cost"] = prod.material_cost + prod.labor_cost
    prod["margin"] = prod.unit_price_usd - prod.unit_cost
    return prod.merge(
        lines[["line_id", "line_name"]], left_on="primary_line_id", right_on="line_id"
    )


def waste_detail(raw: str = "data/raw") -> dict:
    prod = _unit_costs(raw)
    log = pd.read_csv(f"{raw}/fact_production_log.csv", parse_dates=["date"])
    years = log.date.nunique() / 365
    g = log.merge(prod[["product_id", "product_name", "line_name", "unit_cost"]], on="product_id")
    g["started"] = g.produced_units + g.scrap_units
    g["scrap_cost"] = g.scrap_units * g.unit_cost
    by_line = g.groupby("line_name").agg(
        started=("started", "sum"), scrap=("scrap_units", "sum"), cost=("scrap_cost", "sum")
    )
    by_prod = g.groupby(["product_name", "line_name"]).agg(
        started=("started", "sum"), scrap=("scrap_units", "sum"), cost=("scrap_cost", "sum")
    )
    by_prod["rate"] = by_prod.scrap / by_prod.started
    by_prod["target_save"] = np.where(
        by_prod.rate > SCRAP_TARGET,
        (by_prod.rate - SCRAP_TARGET) * by_prod.started * (by_prod.cost / by_prod.scrap),
        0.0,
    )
    mo = (
        g.assign(m=g.date.dt.strftime("%Y-%m"))
        .groupby("m")
        .agg(started=("started", "sum"), scrap=("scrap_units", "sum"), cost=("scrap_cost", "sum"))
    )
    top = by_prod.sort_values("cost", ascending=False).head(10).reset_index()
    total_cost = float(g.scrap_cost.sum())
    revenue = float(
        pd.read_csv(f"{raw}/fact_sales_history.csv", parse_dates=["date"])
        .query("date.dt.year == 2025")
        .revenue_usd.sum()
    )
    return {
        "lines": [
            {
                "line_name": k,
                "rate": float(r.scrap / r.started),
                "cost_year": float(r.cost / years),
                "units_year": float(r.scrap / years),
            }
            for k, r in by_line.iterrows()
        ],
        "top_products": [
            {
                "product_name": r.product_name,
                "line_name": r.line_name,
                "rate": float(r.rate),
                "cost_year": float(r.cost / years),
                "save_year": float(r.target_save / years),
            }
            for r in top.itertuples()
        ],
        "months": list(mo.index),
        "monthly_rate": [float(x) for x in (mo.scrap / mo.started).values],
        "monthly_cost": [float(x) for x in mo.cost.values],
        "kpi": {
            "cost_year": total_cost / years,
            "rate": float(g.scrap_units.sum() / g.started.sum()),
            "units_year": float(g.scrap_units.sum() / years),
            "pct_revenue": total_cost / years / revenue,
            "save_year": float(by_prod.target_save.sum() / years),
            "target": SCRAP_TARGET,
            "worst_line": max(by_line.index, key=lambda k: by_line.cost[k] / by_line.started[k]),
            "n_above_target": int((by_prod.rate > SCRAP_TARGET).sum()),
            "n_products": len(by_prod),
        },
    }


def plant_economics(
    raw: str = "data/raw", db: str = "data/processed/planning_warehouse.db"
) -> dict:
    prod = _unit_costs(raw)
    sales = pd.read_csv(f"{raw}/fact_sales_history.csv", parse_dates=["date"])
    emp = pd.read_csv(f"{raw}/dim_employee.csv")
    lines = pd.read_csv(f"{raw}/dim_production_line.csv").set_index("line_id")
    log = pd.read_csv(f"{raw}/fact_production_log.csv", parse_dates=["date"])
    y = sales[sales.date.dt.year == 2025].merge(
        prod[["product_id", "material_cost", "primary_line_id", "line_name"]], on="product_id"
    )
    y["mat"] = y.units_sold * y.material_cost
    payroll = emp.monthly_wage_usd.sum() * 12
    by_line_pay = (
        emp.dropna(subset=["production_line_id"])
        .groupby("production_line_id")
        .monthly_wage_usd.sum()
        * 12
    )
    rows = []
    l25 = log[log.date.dt.year == 2025]
    for lid, g in y.groupby("primary_line_id"):
        mh = float(
            ((l25[l25.line_id == lid].produced_units + l25[l25.line_id == lid].scrap_units).sum())
            / lines.capacity_units_per_hour[lid]
        )
        contrib = float(g.revenue_usd.sum() - g.mat.sum() - by_line_pay[lid])
        rows.append(
            {
                "line_name": g.line_name.iloc[0],
                "revenue": float(g.revenue_usd.sum()),
                "materials": float(g.mat.sum()),
                "labor": float(by_line_pay[lid]),
                "contribution": contrib,
                "margin": contrib / float(g.revenue_usd.sum()),
                "per_machine_hour": contrib / mh,
                "units": float(g.units_sold.sum()),
            }
        )
    mo = (
        y.assign(m=y.date.dt.strftime("%Y-%m"))
        .groupby(["m", "line_name"])
        .revenue_usd.sum()
        .unstack(fill_value=0)
    )
    rev = float(y.revenue_usd.sum())
    mat = float(y.mat.sum())
    other_pay = float(payroll - by_line_pay.sum())
    return {
        "lines": rows,
        "months": list(mo.index),
        "monthly_revenue": {c: [float(v) for v in mo[c].values] for c in mo.columns},
        "kpi": {
            "revenue": rev,
            "materials": mat,
            "payroll": float(payroll),
            "direct_payroll": float(by_line_pay.sum()),
            "other_payroll": other_pay,
            "contribution": rev - mat - float(payroll),
            "gross_margin": (rev - mat) / rev,
            "contribution_margin": (rev - mat - float(payroll)) / rev,
            "materials_pct": mat / rev,
            "payroll_pct": float(payroll) / rev,
            "employees": int(len(emp)),
            "revenue_per_employee": rev / len(emp),
            "units": float(y.units_sold.sum()),
            "cash_per_ccc_day": rev / 365,
            "year": 2025,
        },
    }


def value_ledger(bw: dict, waste: dict, econ: dict, lines_total: dict) -> list[dict]:
    """Annualised opportunities the engine finds (one-off cash vs recurring savings)."""
    k = bw["kpi"]
    years = max(len(bw["timeline"]) / 52, 1e-9)
    return [
        {"key": "inventory_cash", "kind": "cash", "value": k["freed_cash"]},
        {
            "key": "inventory_cost",
            "kind": "annual",
            "value": (k["holding_cost"] - k["managed_holding_cost"]) / years,
        },
        {"key": "scrap", "kind": "annual", "value": waste["kpi"]["save_year"]},
        {"key": "bottleneck", "kind": "annual", "value": lines_total["bottleneck_margin"]},
        {"key": "ccc_day", "kind": "cash", "value": econ["kpi"]["cash_per_ccc_day"] * 5},
    ]
