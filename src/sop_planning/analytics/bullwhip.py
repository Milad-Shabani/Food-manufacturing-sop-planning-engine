"""Bullwhip effect on raw-material purchasing, and its cash cost.

Every purchase order starts as a *true need* (the quantity the warehouse
calculated from consumption, e.g. 100 kg). Each management layer then adds
its own "just in case" tolerance on top of the previous one, and a supplier
deal adds a final uplift (the 100 kg need becomes a 140 kg request and a
170 kg purchase). The extra stock sits in the warehouse as tied-up cash.

`simulate_chain` builds that decision log for every PO in
`fact_purchase_orders` (seeded, reproducible) and a **managed** counterfactual
with ONE data-driven tolerance per material (service-level z x consumption
volatility) and a deal gate (accept the uplift only if the discount beats the
cost of capital on the extra stock). `summarize` turns both into dashboard KPIs.

The layer percentages are documented assumptions (see PARAMS), not measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS = dict(
    t_supervisor=(0.07, 0.13),  # first-line manager's tolerance
    t_manager=(0.07, 0.15),  # next manager adds another
    t_procurement=(0.10, 0.28),  # buyer's own pad before negotiating
    layer_prob=(0.95, 0.85, 0.90),  # chance each layer adds its tolerance on a given PO
    deal_prob=0.50,  # share of POs bought on a "good offer"
    deal_prob_short_terms=0.20,  # suppliers paid in <=15 days rarely discount
    deal_uplift=(0.10, 0.30),  # extra quantity bought to get the offer
    discount_per_uplift=0.06,  # price discount earned per 1.0 of uplift
    discount_cap=0.05,
    capital_cost_annual=0.55,  # cost of cash under a liquidity squeeze
    service_z=1.65,  # 95% service level for the managed tolerance
    managed_deal_cap=0.10,  # managed policy: max deal uplift vs need
)


def _daily_consumption(raw: str) -> pd.DataFrame:
    inv = pd.read_csv(f"{raw}/fact_inventory_transactions.csv", parse_dates=["date"])
    c = inv[inv.transaction_type == "Consumption"].assign(q=lambda d: -d.quantity)
    g = c.groupby("material_id")["q"]
    return pd.DataFrame({"daily": g.mean(), "sigma": g.std()}).reset_index()


def simulate_chain(raw: str = "data/raw", seed: int = 42, **over) -> pd.DataFrame:
    p = {**PARAMS, **over}
    rng = np.random.default_rng(seed)
    po = pd.read_csv(
        f"{raw}/fact_purchase_orders.csv",
        parse_dates=["order_date", "actual_delivery_date", "expected_delivery_date"],
    )
    mat = pd.read_csv(f"{raw}/dim_raw_material.csv")[
        ["material_id", "material_name", "unit_of_measure"]
    ]
    sup = pd.read_csv(f"{raw}/dim_supplier.csv")[["supplier_id", "payment_terms_days"]]
    d = (
        po.merge(mat, on="material_id")
        .merge(sup, on="supplier_id")
        .merge(_daily_consumption(raw), on="material_id")
    )
    n = len(d)

    def u(lo_hi):
        return rng.uniform(*lo_hi, n)

    def on(i):
        return rng.random(n) < p["layer_prob"][i]

    d["need_qty"] = d.quantity
    d["supervisor_qty"] = d.need_qty * (1 + u(p["t_supervisor"]) * on(0))
    d["manager_qty"] = d.supervisor_qty * (1 + u(p["t_manager"]) * on(1))
    d["procurement_qty"] = d.manager_qty * (1 + u(p["t_procurement"]) * on(2))
    prob = np.where(d.payment_terms_days <= 15, p["deal_prob_short_terms"], p["deal_prob"])
    d["deal"] = rng.random(n) < prob
    d["deal_uplift_pct"] = np.where(d.deal, u(p["deal_uplift"]), 0.0)
    d["purchase_qty"] = d.procurement_qty * (1 + d.deal_uplift_pct)
    d["discount_pct"] = np.minimum(p["discount_cap"], p["discount_per_uplift"] * d.deal_uplift_pct)
    uc, cc = d.unit_cost_usd, p["capital_cost_annual"]
    d["excess_qty"] = d.purchase_qty - d.need_qty
    d["excess_value_usd"] = d.excess_qty * uc
    d["hold_days"] = (d.need_qty + d.excess_qty / 2) / d.daily  # FIFO: excess burns after the need
    d["holding_cost_usd"] = d.excess_value_usd * cc * d.hold_days / 365
    slice_qty = d.procurement_qty * d.deal_uplift_pct
    slice_hold = (d.need_qty + (d.procurement_qty - d.need_qty) + slice_qty / 2) / d.daily
    d["deal_net_usd"] = np.where(
        d.deal, d.discount_pct * uc * d.purchase_qty - slice_qty * uc * cc * slice_hold / 365, 0.0
    )
    # ---- managed policy: one data-driven tolerance + deal gate ----
    d["tolerance_pct"] = p["service_z"] * d.sigma * np.sqrt(d.need_qty / d.daily) / d.need_qty
    base = d.need_qty * (1 + d.tolerance_pct)
    sl = np.where(d.deal, np.minimum(slice_qty, p["managed_deal_cap"] * d.need_qty), 0.0)
    disc = np.minimum(p["discount_cap"], p["discount_per_uplift"] * sl / base)
    net = disc * uc * (base + sl) - sl * uc * cc * ((base + sl / 2) / d.daily) / 365
    d["managed_qty"] = base + np.where(net > 0, sl, 0.0)
    d["managed_excess_value_usd"] = (d.managed_qty - d.need_qty) * uc
    d["managed_hold_days"] = (d.need_qty + (d.managed_qty - d.need_qty) / 2) / d.daily
    d["managed_holding_cost_usd"] = d.managed_excess_value_usd * cc * d.managed_hold_days / 365
    d["receipt_date"] = d.actual_delivery_date.fillna(d.expected_delivery_date)
    return d


def _tied_series(d: pd.DataFrame, col: str, days: pd.DatetimeIndex) -> np.ndarray:
    """Cash tied in excess stock per day: flat while the need is consumed, then burns down."""
    out = np.zeros(len(days))
    t0 = days[0]
    for r in d.itertuples():
        i = (r.receipt_date - t0).days
        keep = r.need_qty / r.daily
        burn = max(
            (
                (r.purchase_qty - r.need_qty) / r.daily
                if col == "excess_value_usd"
                else (r.managed_qty - r.need_qty) / r.daily
            ),
            1e-9,
        )
        v = getattr(r, col)
        k = np.arange(len(days)) - i
        out += np.where(
            (k >= 0) & (k < keep),
            v,
            np.where((k >= keep) & (k < keep + burn), v * (1 - (k - keep) / burn), 0.0),
        )
    return out


def summarize(raw: str = "data/raw", seed: int = 42) -> dict:
    d = simulate_chain(raw, seed)
    uc = d.unit_cost_usd
    w = d.need_qty * uc

    def ratio(c):
        return float((d[c] * uc).sum() / w.sum())

    days = pd.date_range(
        d.receipt_date.min(), d.receipt_date.max() + pd.Timedelta(days=60), freq="D"
    )
    act, man = _tied_series(d, "excess_value_usd", days), _tied_series(
        d, "managed_excess_value_usd", days
    )
    s = pd.DataFrame({"a": act, "m": man}, index=days).resample("W-MON").mean()
    s = s.iloc[2:-2]  # drop ramp-up / ramp-down weeks
    inv = pd.read_csv(f"{raw}/fact_inventory_transactions.csv", parse_dates=["date"])
    inv["cat"] = inv.transaction_type
    cons = inv[inv.cat == "Consumption"].assign(q=lambda x: -x.quantity)
    mats = pd.read_csv(f"{raw}/dim_raw_material.csv")[["material_id", "unit_cost_usd"]]
    snap = inv[inv.cat == "OnHandSnapshot"].merge(mats, on="material_id")
    avg_onhand_val = float((snap.quantity * snap.unit_cost_usd).groupby(snap.date).sum().mean())
    cons_val = cons.merge(mats, on="material_id")
    annual_cons_val = float(
        (cons_val.q * cons_val.unit_cost_usd).sum() / (cons.date.nunique() / 365)
    )
    late = d.actual_delivery_date > d.expected_delivery_date
    mat_g = (
        d.groupby(["material_name", "unit_of_measure"], as_index=False)
        .agg(actual=("excess_value_usd", "sum"), managed=("managed_excess_value_usd", "sum"))
        .sort_values("actual", ascending=False)
    )
    top = d.sort_values("excess_value_usd", ascending=False).head(8)

    def wk(x):
        return str(x.date())

    layers = {
        "tolerance_layers": float(((d.manager_qty - d.need_qty) * uc).sum()),
        "procurement_pad": float(((d.procurement_qty - d.manager_qty) * uc).sum()),
        "deal_uplift": float(((d.purchase_qty - d.procurement_qty) * uc).sum()),
    }
    return {
        "stack": {
            k: ratio(c) * 100
            for k, c in [
                ("need", "need_qty"),
                ("supervisor", "supervisor_qty"),
                ("manager", "manager_qty"),
                ("procurement", "procurement_qty"),
                ("purchase", "purchase_qty"),
                ("managed", "managed_qty"),
            ]
        },
        "layers": layers,
        "timeline": [
            {"week": wk(i), "actual": float(r.a), "managed": float(r.m)} for i, r in s.iterrows()
        ],
        "by_material": mat_g.head(8).to_dict("records"),
        "top_pos": [
            dict(
                po_id=r.po_id,
                material_name=r.material_name,
                unit=r.unit_of_measure,
                need=r.need_qty,
                purchase=r.purchase_qty,
                excess_value=r.excess_value_usd,
                hold_days=r.hold_days,
                deal=bool(r.deal),
                deal_net=r.deal_net_usd,
            )
            for r in top.itertuples()
        ],
        "kpi": {
            "amplification": ratio("purchase_qty"),
            "deal_po_amplification": float(
                (d.purchase_qty[d.deal] * uc[d.deal]).sum()
                / (d.need_qty[d.deal] * uc[d.deal]).sum()
            ),
            "managed_ratio": ratio("managed_qty"),
            "excess_value": float(d.excess_value_usd.sum()),
            "purchase_value": float((d.purchase_qty * uc).sum()),
            "need_value": float(w.sum()),
            "avg_tied": float(s.a.mean()),
            "peak_tied": float(s.a.max()),
            "managed_avg_tied": float(s.m.mean()),
            "freed_cash": float(s.a.mean() - s.m.mean()),
            "holding_cost": float(d.holding_cost_usd.sum()),
            "managed_holding_cost": float(d.managed_holding_cost_usd.sum()),
            "n_pos": int(len(d)),
            "overbuy20_pct": float((d.purchase_qty / d.need_qty > 1.2).mean()),
            "n_deals": int(d.deal.sum()),
            "deals_positive_pct": float((d.deal_net_usd[d.deal] > 0).mean()),
            "deal_net_total": float(d.deal_net_usd.sum()),
            "avg_tolerance_managed": float(d.tolerance_pct.mean()),
            "stack_tolerance": float(ratio("procurement_qty") - 1),
            "on_time_pct": float(1 - late.mean()),
            "avg_delay_days": float(
                (d.actual_delivery_date - d.expected_delivery_date).dt.days.clip(lower=0).mean()
            ),
            "inventory_turns": annual_cons_val / avg_onhand_val,
            "days_cover": avg_onhand_val / (annual_cons_val / 365),
        },
    }


def write_chain(raw: str = "data/raw", seed: int = 42) -> pd.DataFrame:
    d = simulate_chain(raw, seed)
    cols = [
        "po_id",
        "material_id",
        "supplier_id",
        "order_date",
        "need_qty",
        "supervisor_qty",
        "manager_qty",
        "procurement_qty",
        "deal",
        "deal_uplift_pct",
        "discount_pct",
        "purchase_qty",
        "excess_qty",
        "excess_value_usd",
        "hold_days",
        "holding_cost_usd",
        "deal_net_usd",
        "tolerance_pct",
        "managed_qty",
        "managed_excess_value_usd",
    ]
    out = d[cols].copy()
    out["order_date"] = out.order_date.dt.strftime("%Y-%m-%d")
    out = out.round(4)
    out.to_csv(f"{raw}/fact_replenishment_chain.csv", index=False)
    return out
