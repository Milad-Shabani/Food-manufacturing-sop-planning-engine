"""Cash Conversion Cycle (CCC) and its components: DIO, DSO, DPO.

CCC = DIO + DSO - DPO

- **DIO** (Days Inventory Outstanding): how long raw materials sit in the
  warehouse before being consumed in production. Computed directly from
  the simulated inventory ledger (`fact_inventory_transactions`) — real
  on-hand values and real consumption, not an assumption.
- **DSO** (Days Sales Outstanding): how long it takes to collect cash from
  customers after a sale. Simulated from an accounts-receivable ledger:
  each day's revenue is "invoiced" and collected `payment_terms_days`
  (per region, from `dim_region`) later, with a little jitter.
- **DPO** (Days Payable Outstanding): how long the company takes to pay its
  suppliers after receiving goods. Simulated the same way from an
  accounts-payable ledger using `dim_supplier.payment_terms_days`.

All three are computed as trailing-window averages
(`avg balance / period value * window_days`) rather than single point-in-time
snapshots, which is both the standard finance definition and what makes a
sensible time series to chart.

This ties the CCC directly to the same supply and production levers the
rest of the engine plans around: a tighter safety-stock policy lowers DIO,
faster-paying customers or slower-paying suppliers lower/raise DSO and DPO
— the same knobs `datagen` and `planning` already turn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _shifted_ledger(
    events: pd.DataFrame, amount_col: str, date_col: str, terms_col: str, group_col: str, seed: int
) -> pd.DataFrame:
    """Build a running balance for an AR- or AP-style ledger: an amount is
    recognized on `date_col` and settled `terms_col` days later (±2 days of
    jitter), per group (region or supplier). Returns one row per date with
    the recognized amount, settled amount, and running balance.
    """
    rng = np.random.default_rng(seed)
    df = events.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    jitter = rng.integers(-2, 3, size=len(df))
    df["_settle_date"] = df[date_col] + pd.to_timedelta(df[terms_col] + jitter, unit="D")

    all_dates = pd.date_range(df[date_col].min(), df["_settle_date"].max(), freq="D")

    recognized = df.groupby(date_col)[amount_col].sum().reindex(all_dates, fill_value=0.0)
    settled = df.groupby("_settle_date")[amount_col].sum().reindex(all_dates, fill_value=0.0)

    balance = (recognized.cumsum() - settled.cumsum()).clip(lower=0)

    return pd.DataFrame(
        {
            "date": all_dates,
            "recognized": recognized.values,
            "settled": settled.values,
            "balance": balance.values,
        }
    )


def build_receivables_ledger(
    sales_history: pd.DataFrame, regions: pd.DataFrame, seed: int = 101
) -> pd.DataFrame:
    daily_revenue = sales_history.merge(
        regions[["region_id", "payment_terms_days"]], on="region_id"
    )
    daily_revenue = daily_revenue.groupby(["date", "payment_terms_days"], as_index=False)[
        "revenue_irr"
    ].sum()
    daily_revenue = daily_revenue.rename(columns={"revenue_irr": "amount"})
    return _shifted_ledger(
        daily_revenue, "amount", "date", "payment_terms_days", "payment_terms_days", seed
    )


def build_payables_ledger(
    purchase_orders: pd.DataFrame, suppliers: pd.DataFrame, seed: int = 102
) -> pd.DataFrame:
    po = purchase_orders.merge(suppliers[["supplier_id", "payment_terms_days"]], on="supplier_id")
    po = po.copy()
    po["amount"] = po["quantity"] * po["unit_cost_irr"]
    # The payable is recognized when goods are received, not when ordered.
    po = po.groupby(["actual_delivery_date", "payment_terms_days"], as_index=False)["amount"].sum()
    return _shifted_ledger(
        po, "amount", "actual_delivery_date", "payment_terms_days", "payment_terms_days", seed
    )


def build_inventory_value_series(
    inventory: pd.DataFrame, raw_materials: pd.DataFrame
) -> pd.DataFrame:
    """Daily raw-material inventory value (on-hand) and consumption value,
    valued at each material's unit cost."""
    unit_cost = dict(zip(raw_materials["material_id"], raw_materials["unit_cost_irr"]))
    df = inventory.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df.apply(lambda r: r["quantity"] * unit_cost.get(r["material_id"], 0.0), axis=1)

    on_hand_value = (
        df[df["transaction_type"] == "OnHandSnapshot"]
        .groupby("date")["value"]
        .sum()
        .rename("inventory_value")
    )
    consumption_value = (
        df[df["transaction_type"] == "Consumption"]
        .groupby("date")["value"]
        .sum()
        .abs()
        .rename("consumption_value")
    )
    return pd.concat([on_hand_value, consumption_value], axis=1).fillna(0.0).reset_index()


def compute_ccc_timeseries(
    sales_history: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    inventory: pd.DataFrame,
    regions: pd.DataFrame,
    suppliers: pd.DataFrame,
    raw_materials: pd.DataFrame,
    window_days: int = 90,
    freq: str = "W-SAT",
) -> pd.DataFrame:
    """Weekly DIO / DSO / DPO / CCC, each a trailing-`window_days` average."""
    ar = build_receivables_ledger(sales_history, regions).set_index("date")
    ap = build_payables_ledger(purchase_orders, suppliers).set_index("date")
    inv = build_inventory_value_series(inventory, raw_materials).set_index("date")

    revenue_by_day = sales_history.copy()
    revenue_by_day["date"] = pd.to_datetime(revenue_by_day["date"])
    revenue_by_day = revenue_by_day.groupby("date")["revenue_irr"].sum()

    purchase_value_by_day = purchase_orders.copy()
    purchase_value_by_day["date"] = pd.to_datetime(purchase_value_by_day["actual_delivery_date"])
    purchase_value_by_day["amount"] = (
        purchase_value_by_day["quantity"] * purchase_value_by_day["unit_cost_irr"]
    )
    purchase_value_by_day = purchase_value_by_day.groupby("date")["amount"].sum()

    idx = pd.date_range(
        min(ar.index.min(), ap.index.min(), inv.index.min()),
        max(ar.index.max(), ap.index.max(), inv.index.max()),
        freq="D",
    )
    ar_bal = ar["balance"].reindex(idx, fill_value=0.0)
    ap_bal = ap["balance"].reindex(idx, fill_value=0.0)
    inv_val = inv["inventory_value"].reindex(idx, fill_value=0.0)
    cons_val = inv["consumption_value"].reindex(idx, fill_value=0.0)
    revenue = revenue_by_day.reindex(idx, fill_value=0.0)
    purchases = purchase_value_by_day.reindex(idx, fill_value=0.0)

    # Cap reporting to the actual span of sales/purchase activity. The AR/AP
    # ledgers extend past that (a sale on the last day still has an open
    # invoice for `payment_terms_days` afterward) — including those tail
    # weeks would divide a shrinking trailing revenue/purchase sum by a
    # balance that's structurally still owed, producing runaway ratios that
    # reflect the simulation's edge, not the business.
    last_data_date = min(revenue_by_day.index.max(), purchase_value_by_day.index.max())
    week_ends = pd.date_range(idx.min(), last_data_date, freq=freq)
    # Skip the ramp-up period: the AR/AP ledgers start at zero balance and
    # need a full payment-term cycle (the slowest one, e.g. 120-day export
    # terms) plus the trailing window itself before they reach a realistic
    # steady state. Reporting earlier than that would show an artificially
    # improving trend that's really just the ledger filling up.
    max_terms = max(regions["payment_terms_days"].max(), suppliers["payment_terms_days"].max())
    warm_up_days = window_days + int(max_terms) + 3
    earliest_reportable = idx.min() + pd.Timedelta(days=warm_up_days)

    rows = []
    for we in week_ends:
        if we < earliest_reportable:
            continue
        start = we - pd.Timedelta(days=window_days - 1)
        window = pd.date_range(start, we)
        n = len(window)

        avg_ar = ar_bal.reindex(window).mean()
        avg_ap = ap_bal.reindex(window).mean()
        avg_inv = inv_val.reindex(window).mean()
        rev_sum = revenue.reindex(window).sum()
        pur_sum = purchases.reindex(window).sum()
        cons_sum = cons_val.reindex(window).sum()

        dso = (avg_ar / rev_sum * n) if rev_sum else np.nan
        dpo = (avg_ap / pur_sum * n) if pur_sum else np.nan
        dio = (avg_inv / cons_sum * n) if cons_sum else np.nan
        ccc = dio + dso - dpo if pd.notna([dio, dso, dpo]).all() else np.nan

        rows.append(
            {
                "week_start": we - pd.Timedelta(days=6),
                "dio": round(dio, 1) if pd.notna(dio) else None,
                "dso": round(dso, 1) if pd.notna(dso) else None,
                "dpo": round(dpo, 1) if pd.notna(dpo) else None,
                "ccc": round(ccc, 1) if pd.notna(ccc) else None,
            }
        )

    return pd.DataFrame(rows)
