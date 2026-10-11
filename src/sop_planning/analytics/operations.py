"""Production-line and warehouse detail for the dashboard.

Lines: OEE (availability x performance x quality) from the production log, scrap,
downtime, changeover loss, workforce attendance, and the planned utilization of
machine hours and labor hours per line-week (same capacity model as the LP).
Warehouse: stock value, days of cover vs the safety-stock target, ABC class,
turns, time spent below safety stock, and inbound supplier reliability.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from sop_planning.config import load_settings

HOURS_PER_SHIFT = 8


def _ym(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m")


def line_detail(raw: str = "data/raw", db: str = "data/processed/planning_warehouse.db") -> dict:
    cfg = load_settings()
    oee_plan = cfg.oee
    ops = cfg.operators_per_running_hour
    lines = pd.read_csv(f"{raw}/dim_production_line.csv")
    log = pd.read_csv(f"{raw}/fact_production_log.csv", parse_dates=["date"])
    wf = pd.read_csv(f"{raw}/fact_workforce_availability.csv", parse_dates=["date"])
    emp = pd.read_csv(f"{raw}/dim_employee.csv")
    plan = pd.read_sql("select * from fact_production_plan", sqlite3.connect(db))

    out_lines, trend, heat = [], {}, []
    days = log.date.nunique()
    weeks = sorted(plan.week_start.unique())
    wf["week_start"] = wf.date.dt.to_period("W-SAT").apply(lambda p: p.start_time)
    wk_labor = wf.groupby(["line_id", "week_start"]).present_headcount.sum() * HOURS_PER_SHIFT
    avg_labor = wk_labor.groupby(level=0).mean()
    for ln in lines.itertuples():
        g = log[log.line_id == ln.line_id]
        avail_min = ln.shifts_per_day * HOURS_PER_SHIFT * 60
        daily = g.groupby("date").agg(dt=("downtime_minutes", "sum"), n=("product_id", "nunique"))
        a = 1 - daily.dt.mean() / avail_min
        perf = g.produced_units.sum() / g.planned_units.sum()
        qual = 1 - g.scrap_units.sum() / g.produced_units.sum()
        co_loss = ((daily.n - 1).clip(lower=0) * ln.changeover_minutes).mean() / avail_min
        m = g.assign(m=_ym(g.date)).groupby("m")
        mo = 1 - m.downtime_minutes.sum() / (
            avail_min * g.assign(m=_ym(g.date)).groupby("m").date.nunique()
        )
        mo = (
            mo
            * (m.produced_units.sum() / m.planned_units.sum())
            * (1 - m.scrap_units.sum() / m.produced_units.sum())
        )
        trend[ln.line_name] = [round(float(v), 4) for v in mo.values]
        months = list(mo.index)
        w = wf[wf.line_id == ln.line_id]
        pl = plan[plan.line_id == ln.line_id].groupby("week_start").allocated_units.sum()
        mach_av = ln.shifts_per_day * HOURS_PER_SHIFT * 7 * oee_plan
        lab_per_unit = ops[ln.line_id] / ln.capacity_units_per_hour
        mu, lu = [], []
        for wk in weeks:
            used = float(pl.get(wk, 0.0))
            mu.append(used / ln.capacity_units_per_hour / mach_av)
            lu.append(
                used
                * lab_per_unit
                / float(wk_labor.get((ln.line_id, pd.Timestamp(wk)), avg_labor[ln.line_id]))
            )
        util = np.maximum(mu, lu)
        heat.append(
            {
                "line_name": ln.line_name,
                "util": [round(float(x), 4) for x in util],
                "binding": ["labor" if l > m_ else "machine" for m_, l in zip(mu, lu)],
            }
        )
        out_lines.append(
            {
                "line_id": ln.line_id,
                "line_name": ln.line_name,
                "capacity": int(ln.capacity_units_per_hour),
                "shifts": int(ln.shifts_per_day),
                "changeover_min": int(ln.changeover_minutes),
                "headcount": int(
                    (
                        (emp.production_line_id == ln.line_id) & (emp.role == "Production Operator")
                    ).sum()
                ),
                "age_years": round(2026 - pd.Timestamp(ln.commissioned_date).year, 0),
                "availability": float(a),
                "performance": float(perf),
                "quality": float(qual),
                "oee": float(a * perf * qual),
                "scrap_pct": float(g.scrap_units.sum() / g.produced_units.sum()),
                "downtime_per_day": float(daily.dt.mean()),
                "changeover_loss": float(co_loss),
                "attendance": float(w.present_headcount.sum() / w.scheduled_headcount.sum()),
                "avg_util": float(np.mean(util)),
                "peak_util": float(np.max(util)),
                "peak_binding": heat[-1]["binding"][int(np.argmax(util))],
            }
        )
    shift = wf.groupby("shift").agg(
        p=("present_headcount", "sum"), s=("scheduled_headcount", "sum")
    )
    mt = (
        wf.assign(m=_ym(wf.date))
        .groupby("m")
        .agg(p=("present_headcount", "sum"), s=("scheduled_headcount", "sum"))
    )
    return {
        "lines": out_lines,
        "months": months,
        "oee_trend": trend,
        "weeks": [str(pd.Timestamp(x).date()) for x in weeks],
        "heat": heat,
        "attendance_shift": [
            {"shift": k, "attendance": float(r.p / r.s)} for k, r in shift.iterrows()
        ],
        "attendance_trend": [float(x) for x in (mt.p / mt.s).values],
        "attendance_months": list(mt.index),
        "kpi": {
            "avg_oee": float(np.average([x["oee"] for x in out_lines])),
            "scrap_pct": float(log.scrap_units.sum() / log.produced_units.sum()),
            "downtime_per_day": float(np.mean([x["downtime_per_day"] for x in out_lines])),
            "attendance": float(wf.present_headcount.sum() / wf.scheduled_headcount.sum()),
            "changeover_loss": float(np.mean([x["changeover_loss"] for x in out_lines])),
            "peak_line": max(out_lines, key=lambda x: x["peak_util"])["line_name"],
            "peak_util": max(x["peak_util"] for x in out_lines),
            "operators": int(sum(x["headcount"] for x in out_lines)),
            "days": int(days),
        },
    }


def warehouse_detail(raw: str = "data/raw") -> dict:
    inv = pd.read_csv(f"{raw}/fact_inventory_transactions.csv", parse_dates=["date"])
    mat = pd.read_csv(f"{raw}/dim_raw_material.csv")
    po = pd.read_csv(
        f"{raw}/fact_purchase_orders.csv",
        parse_dates=["expected_delivery_date", "actual_delivery_date"],
    )
    sup = pd.read_csv(f"{raw}/dim_supplier.csv")
    snap = inv[inv.transaction_type == "OnHandSnapshot"].pivot(
        index="date", columns="material_id", values="quantity"
    )
    cons = -inv[inv.transaction_type == "Consumption"].pivot(
        index="date", columns="material_id", values="quantity"
    )
    rec = inv[inv.transaction_type == "Receipt"]
    mat = mat.set_index("material_id")
    daily = cons.mean()
    cover = snap / daily  # days of cover, every day
    value = snap * mat.unit_cost_usd
    cons_val = (cons * mat.unit_cost_usd).sum()
    order = cons_val.sort_values(ascending=False)
    cum = order.cumsum() / order.sum()
    abc = {
        m: (
            "A"
            if c - order[m] / order.sum() < 0.8
            else "B" if c - order[m] / order.sum() < 0.95 else "C"
        )
        for m, c in cum.items()
    }
    years = (snap.index.max() - snap.index.min()).days / 365
    rows = []
    for m, r in mat.iterrows():
        now = float(snap[m].iloc[-1])
        dc = now / daily[m]
        status = (
            "below"
            if dc < r.safety_stock_days
            else "excess" if dc > 2 * (r.lead_time_days + r.safety_stock_days) else "ok"
        )
        rows.append(
            {
                "material_id": m,
                "material_name": r.material_name,
                "unit": r.unit_of_measure,
                "abc": abc[m],
                "on_hand": now,
                "value": now * r.unit_cost_usd,
                "days_cover": float(dc),
                "safety_days": int(r.safety_stock_days),
                "lead_days": int(r.lead_time_days),
                "turns": float(cons_val[m] / years / value[m].mean()),
                "below_pct": float((cover[m] < r.safety_stock_days).mean()),
                "stockout_days": int((snap[m] <= 0).sum()),
                "share": float(cons_val[m] / cons_val.sum()),
                "status": status,
            }
        )
    rows.sort(key=lambda x: -x["share"])
    vm = value.sum(axis=1).resample("MS").mean()
    po["late"] = (po.actual_delivery_date - po.expected_delivery_date).dt.days.clip(lower=0)
    po["spend"] = po.quantity * po.unit_cost_usd
    sp = po.groupby("supplier_id").agg(
        n=("po_id", "count"),
        spend=("spend", "sum"),
        on_time=("late", lambda s: (s == 0).mean()),
        delay=("late", "mean"),
    )
    sp = sp.join(
        sup.set_index("supplier_id")[["supplier_name", "reliability_score", "payment_terms_days"]]
    )
    return {
        "materials": rows,
        "value_trend": [{"month": str(i.date())[:7], "value": float(v)} for i, v in vm.items()],
        "pareto": [
            {
                "material_name": mat.material_name[m],
                "share": float(order[m] / order.sum()),
                "cum": float(cum[m]),
            }
            for m in order.index
        ],
        "suppliers": sorted(
            [
                {
                    "supplier": r.supplier_name,
                    "n": int(r.n),
                    "spend_share": float(r.spend / sp.spend.sum()),
                    "on_time": float(r.on_time),
                    "delay": float(r.delay),
                    "terms": int(r.payment_terms_days),
                }
                for _, r in sp.iterrows()
            ],
            key=lambda x: -x["spend_share"],
        ),
        "kpi": {
            "value_now": float(value.iloc[-1].sum()),
            "value_avg": float(value.sum(axis=1).mean()),
            "avg_cover": float(
                np.average([x["days_cover"] for x in rows], weights=[x["share"] for x in rows])
            ),
            "turns": float(cons_val.sum() / years / value.sum(axis=1).mean()),
            "n_below": sum(x["status"] == "below" for x in rows),
            "n_excess": sum(x["status"] == "excess" for x in rows),
            "n_materials": len(rows),
            "n_a": sum(x["abc"] == "A" for x in rows),
            "a_share": float(sum(x["share"] for x in rows if x["abc"] == "A")),
            "receipts": int(len(rec)),
            "avg_receipt_value": float(
                (rec.quantity * rec.material_id.map(mat.unit_cost_usd)).mean()
            ),
            "stockout_days": int(sum(x["stockout_days"] for x in rows)),
        },
    }
