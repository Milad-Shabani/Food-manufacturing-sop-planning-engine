"""Generate the fact tables: sales history, production log, inventory
transactions, purchase orders, and daily workforce availability.

Facts are generated in dependency order (sales -> production -> material
consumption -> purchase orders -> inventory) so later tables are internally
consistent with earlier ones — e.g. a material's consumption on a given day
comes from that day's actual production log via the BOM, not from an
independent random draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .seasonality import seasonal_multiplier, weekday_multiplier

# Baseline daily demand (units/day, company-wide) for each product category
# before seasonality/weekday/trend/noise are applied.
CATEGORY_BASE_DEMAND = {
    "Sponge Cake": 380,
    "Cream Cake": 260,
    "Kolucheh": 900,
    "Wafer": 1500,
    "Cookie": 700,
    "Seasonal": 120,
}

REGION_WEIGHTS = {
    "R01": 0.34,
    "R02": 0.14,
    "R03": 0.13,
    "R04": 0.11,
    "R05": 0.12,
    "R06": 0.10,
    "R07": 0.06,
}


def generate_date_range(start: str = "2024-01-01", end: str = "2025-12-31") -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def generate_sales_history(
    products: pd.DataFrame, regions: pd.DataFrame, dates: pd.DatetimeIndex, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weekday_mult = weekday_multiplier(dates)
    n_days = len(dates)
    trend = 1.0 + np.linspace(0, 0.14, n_days)  # ~14% organic growth over the window

    region_ids = regions["region_id"].tolist()
    region_w = np.array([REGION_WEIGHTS[r] for r in region_ids])
    region_w = region_w / region_w.sum()

    rows = []
    for _, product in products.iterrows():
        base = CATEGORY_BASE_DEMAND[product["category"]] / _products_per_category(
            products, product["category"]
        )
        seas_mult = seasonal_multiplier(
            dates, product["seasonal_peak"] if product["is_seasonal"] else None
        )

        expected = base * seas_mult * weekday_mult * trend
        # Occasional promo days: ~4% of days get a demand bump from a discount.
        promo_days = rng.random(n_days) < 0.04
        expected = expected * np.where(promo_days, 1.35, 1.0)

        noisy_total = rng.normal(expected, np.sqrt(np.maximum(expected, 1)) * 1.4)
        noisy_total = np.clip(np.round(noisy_total), 0, None).astype(int)

        # Split the day's total across regions with a little multinomial noise.
        for i, d in enumerate(dates):
            total = noisy_total[i]
            if total == 0:
                continue
            region_split = rng.multinomial(total, region_w)
            for region_id, qty in zip(region_ids, region_split):
                if qty == 0:
                    continue
                unit_price = product["unit_price_irr"] * (0.9 if promo_days[i] else 1.0)
                rows.append(
                    {
                        "date": d.date(),
                        "product_id": product["product_id"],
                        "region_id": region_id,
                        "units_sold": int(qty),
                        "unit_price_irr": int(unit_price),
                        "revenue_irr": int(qty * unit_price),
                        "is_promo_day": bool(promo_days[i]),
                    }
                )
    return pd.DataFrame(rows)


def _products_per_category(products: pd.DataFrame, category: str) -> int:
    return int((products["category"] == category).sum())


def generate_production_log(
    sales: pd.DataFrame, products: pd.DataFrame, lines: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Derive a production log from sales demand.

    Production targets ~1.06x of same-day demand (a small buffer for
    inventory/spoilage/short shelf life), then applies line capacity limits,
    random downtime, and a small scrap rate — so `produced_units` doesn't
    always exactly equal the plan, which is the point: the planning engine
    later has to work with imperfect execution, not a clean textbook input.
    """
    rng = np.random.default_rng(seed)
    line_capacity = dict(zip(lines["line_id"], lines["capacity_units_per_hour"]))
    line_shifts = dict(zip(lines["line_id"], lines["shifts_per_day"]))
    line_for_product = dict(zip(products["product_id"], products["primary_line_id"]))

    demand_by_day_product = sales.groupby(["date", "product_id"], as_index=False)[
        "units_sold"
    ].sum()
    demand_by_day_product["planned_units"] = np.ceil(
        demand_by_day_product["units_sold"] * 1.06
    ).astype(int)
    demand_by_day_product["line_id"] = demand_by_day_product["product_id"].map(line_for_product)

    rows = []
    for _, r in demand_by_day_product.iterrows():
        hours_available = line_shifts[r["line_id"]] * 8
        max_capacity = int(
            line_capacity[r["line_id"]] * hours_available * 0.55
        )  # line is shared across products
        planned = min(r["planned_units"], max(max_capacity, 1))

        downtime_minutes = int(max(rng.normal(20, 15), 0))
        scrap_rate = np.clip(rng.normal(0.015, 0.006), 0, 0.08)
        produced = int(round(planned * (1 - downtime_minutes / (hours_available * 60) * 0.5)))
        scrap = int(round(produced * scrap_rate))

        rows.append(
            {
                "date": r["date"],
                "line_id": r["line_id"],
                "product_id": r["product_id"],
                "planned_units": int(planned),
                "produced_units": max(produced - scrap, 0),
                "scrap_units": scrap,
                "downtime_minutes": downtime_minutes,
            }
        )
    return pd.DataFrame(rows)


def generate_material_consumption(production: pd.DataFrame, bom: pd.DataFrame) -> pd.DataFrame:
    """Explode the production log through the BOM into daily material use."""
    merged = production.merge(bom, on="product_id", how="left")
    merged["quantity"] = merged["produced_units"] * merged["qty_per_unit"]
    consumption = merged.groupby(["date", "material_id"], as_index=False)["quantity"].sum()
    consumption["transaction_type"] = "Consumption"
    return consumption[["date", "material_id", "transaction_type", "quantity"]]


def generate_purchase_orders_and_inventory(
    consumption: pd.DataFrame,
    raw_materials: pd.DataFrame,
    dates: pd.DatetimeIndex,
    seed: int,
    starting_stock_days: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a reorder-point purchasing policy and the resulting stock ledger.

    For each material: start with `starting_stock_days` of average demand on
    hand, consume daily per `consumption`, and place a purchase order
    whenever projected stock would fall below (lead_time + safety_stock)
    days of average demand. Deliveries arrive after the supplier's lead time
    with a little random jitter (some suppliers run late) so the resulting
    inventory ledger has genuine, occasionally-tight stock levels rather
    than a permanently comfortable buffer.
    """
    rng = np.random.default_rng(seed)
    po_rows: list[dict] = []
    inv_rows: list[dict] = []
    po_counter = 1

    for _, mat in raw_materials.iterrows():
        mat_id = mat["material_id"]
        daily_use = consumption.loc[consumption["material_id"] == mat_id].set_index("date")[
            "quantity"
        ]
        daily_use = daily_use.reindex([d.date() for d in dates], fill_value=0.0)
        avg_daily_use = max(daily_use.mean(), 0.01)

        on_hand = avg_daily_use * starting_stock_days
        reorder_point = avg_daily_use * (mat["lead_time_days"] + mat["safety_stock_days"])
        order_qty = avg_daily_use * (mat["lead_time_days"] + mat["safety_stock_days"]) * 1.5

        pending_deliveries: dict = {}  # date -> qty

        for d in dates:
            dd = d.date()
            delivered_today = pending_deliveries.pop(dd, 0.0)
            if delivered_today:
                on_hand += delivered_today
                inv_rows.append(
                    {
                        "date": dd,
                        "material_id": mat_id,
                        "transaction_type": "Receipt",
                        "quantity": round(delivered_today, 2),
                    }
                )

            used_today = daily_use.get(dd, 0.0)
            on_hand -= used_today

            if used_today:
                inv_rows.append(
                    {
                        "date": dd,
                        "material_id": mat_id,
                        "transaction_type": "Consumption",
                        "quantity": -round(used_today, 2),
                    }
                )

            # Reorder check: is there already an open order? keep it simple —
            # only place a new order if none is currently pending.
            if on_hand < reorder_point and not pending_deliveries:
                lead = int(max(mat["lead_time_days"] + rng.normal(0, 1.2), 1))
                late_chance = 1 - mat_reliability(raw_materials, mat_id)
                if rng.random() < late_chance:
                    lead += int(rng.integers(2, 6))
                delivery_date = (pd.Timestamp(dd) + pd.Timedelta(days=lead)).date()
                pending_deliveries[delivery_date] = (
                    pending_deliveries.get(delivery_date, 0.0) + order_qty
                )

                po_rows.append(
                    {
                        "po_id": f"PO{po_counter:05d}",
                        "material_id": mat_id,
                        "supplier_id": mat["default_supplier_id"],
                        "order_date": dd,
                        "expected_delivery_date": (
                            pd.Timestamp(dd) + pd.Timedelta(days=int(mat["lead_time_days"]))
                        ).date(),
                        "actual_delivery_date": delivery_date,
                        "quantity": round(order_qty, 2),
                        "unit_cost_irr": mat["unit_cost_irr"],
                        "status": "Delivered",
                    }
                )
                po_counter += 1

            inv_rows.append(
                {
                    "date": dd,
                    "material_id": mat_id,
                    "transaction_type": "OnHandSnapshot",
                    "quantity": round(on_hand, 2),
                }
            )

    return pd.DataFrame(po_rows), pd.DataFrame(inv_rows)


def mat_reliability(raw_materials: pd.DataFrame, material_id: str) -> float:
    supplier_reliability = {
        "SUP01": 0.97,
        "SUP02": 0.98,
        "SUP03": 0.95,
        "SUP04": 0.93,
        "SUP05": 0.90,
        "SUP06": 0.96,
        "SUP07": 0.97,
        "SUP08": 0.94,
    }
    supplier_id = raw_materials.loc[
        raw_materials["material_id"] == material_id, "default_supplier_id"
    ].iloc[0]
    return supplier_reliability.get(supplier_id, 0.95)


def generate_workforce_availability(
    employees: pd.DataFrame, lines: pd.DataFrame, dates: pd.DatetimeIndex, seed: int
) -> pd.DataFrame:
    """Daily scheduled vs. present headcount per line, per shift.

    A ~93% average attendance rate with day-to-day noise is enough to make
    the workforce-capacity check in the planning module meaningful — some
    days a line is genuinely short-staffed relative to its production plan.
    """
    rng = np.random.default_rng(seed)
    production_roles = ("Production Operator", "Line Supervisor", "QC Inspector")
    prod_employees = employees[
        employees["role"].isin(production_roles) & employees["production_line_id"].notna()
    ]

    rows = []
    for line_id in lines["line_id"]:
        line_staff = prod_employees[prod_employees["production_line_id"] == line_id]
        for shift in _shifts_for_staff(line_staff):
            scheduled = int((line_staff["shift"] == shift).sum())
            if scheduled == 0:
                continue
            for d in dates:
                present = int(np.clip(rng.binomial(scheduled, 0.93), 0, scheduled))
                rows.append(
                    {
                        "date": d.date(),
                        "line_id": line_id,
                        "shift": shift,
                        "scheduled_headcount": scheduled,
                        "present_headcount": present,
                    }
                )
    return pd.DataFrame(rows)


def _shifts_for_staff(line_staff: pd.DataFrame) -> list[str]:
    return sorted(line_staff["shift"].unique().tolist())
