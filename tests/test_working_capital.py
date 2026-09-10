import numpy as np
import pandas as pd

from sop_planning.analytics.working_capital import (
    build_payables_ledger,
    build_receivables_ledger,
    compute_ccc_timeseries,
)


def _synthetic_sales(n_days=400, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        for region_id in ["R01", "R02"]:
            rows.append(
                {
                    "date": d.date(),
                    "region_id": region_id,
                    "revenue_irr": float(rng.integers(8_000_000, 12_000_000)),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_purchase_orders(n_orders=200, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=380, freq="D")
    rows = []
    for i in range(n_orders):
        d = rng.choice(dates)
        rows.append(
            {
                "po_id": f"PO{i:04d}",
                "supplier_id": rng.choice(["SUP01", "SUP02"]),
                "actual_delivery_date": pd.Timestamp(d).date(),
                "quantity": float(rng.integers(100, 1000)),
                "unit_cost_irr": float(rng.integers(10000, 50000)),
            }
        )
    return pd.DataFrame(rows)


def _regions():
    return pd.DataFrame(
        [
            {"region_id": "R01", "payment_terms_days": 30},
            {"region_id": "R02", "payment_terms_days": 60},
        ]
    )


def _suppliers():
    return pd.DataFrame(
        [
            {"supplier_id": "SUP01", "payment_terms_days": 20},
            {"supplier_id": "SUP02", "payment_terms_days": 40},
        ]
    )


def test_receivables_ledger_balance_is_never_negative():
    ledger = build_receivables_ledger(_synthetic_sales(), _regions())
    assert (ledger["balance"] >= 0).all()


def test_payables_ledger_balance_is_never_negative():
    ledger = build_payables_ledger(_synthetic_purchase_orders(), _suppliers())
    assert (ledger["balance"] >= 0).all()


def test_receivables_ledger_balance_roughly_tracks_payment_terms():
    # With ~10M IRR/day/region flowing in and ~45-day blended terms, the
    # steady-state AR balance should be in the same order of magnitude as
    # (daily revenue * payment terms), not wildly larger or smaller.
    ledger = build_receivables_ledger(_synthetic_sales(), _regions())
    steady_state = ledger[(ledger["date"] > "2024-04-01") & (ledger["date"] < "2024-10-01")]
    daily_revenue = 2 * 10_000_000  # 2 regions, ~10M avg each
    expected_order_of_magnitude = daily_revenue * 45
    assert (
        0.3 * expected_order_of_magnitude
        < steady_state["balance"].mean()
        < 2 * expected_order_of_magnitude
    )


def test_compute_ccc_timeseries_shape_and_positivity():
    sales = _synthetic_sales()
    po = _synthetic_purchase_orders()
    inv = pd.DataFrame(
        [
            {
                "date": d.date(),
                "material_id": "M01",
                "transaction_type": "OnHandSnapshot",
                "quantity": 1000.0,
            }
            for d in pd.date_range("2024-01-01", periods=400, freq="D")
        ]
        + [
            {
                "date": d.date(),
                "material_id": "M01",
                "transaction_type": "Consumption",
                "quantity": -50.0,
            }
            for d in pd.date_range("2024-01-01", periods=400, freq="D")
        ]
    )
    materials = pd.DataFrame([{"material_id": "M01", "unit_cost_irr": 1000.0}])

    ccc = compute_ccc_timeseries(
        sales, po, inv, _regions(), _suppliers(), materials, window_days=60
    )

    assert not ccc.empty
    assert {"dio", "dso", "dpo", "ccc"}.issubset(ccc.columns)
    assert (ccc["dio"] > 0).all()
    assert (ccc["dso"] > 0).all()


def test_ccc_equals_dio_plus_dso_minus_dpo():
    sales = _synthetic_sales()
    po = _synthetic_purchase_orders()
    inv = pd.DataFrame(
        [
            {
                "date": d.date(),
                "material_id": "M01",
                "transaction_type": "OnHandSnapshot",
                "quantity": 1000.0,
            }
            for d in pd.date_range("2024-01-01", periods=400, freq="D")
        ]
        + [
            {
                "date": d.date(),
                "material_id": "M01",
                "transaction_type": "Consumption",
                "quantity": -50.0,
            }
            for d in pd.date_range("2024-01-01", periods=400, freq="D")
        ]
    )
    materials = pd.DataFrame([{"material_id": "M01", "unit_cost_irr": 1000.0}])

    ccc = compute_ccc_timeseries(
        sales, po, inv, _regions(), _suppliers(), materials, window_days=60
    )
    recomputed = ccc["dio"] + ccc["dso"] - ccc["dpo"]
    # each component is independently rounded to 1 decimal before CCC is
    # derived from full precision, so allow a hair of rounding slack
    assert (recomputed - ccc["ccc"]).abs().max() < 0.2
