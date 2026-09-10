"""Material Requirements Planning (MRP).

Explodes the production plan through the BOM into weekly raw-material
requirements, nets that against current on-hand inventory, and recommends
purchase orders — sized and timed so they land before stock would run out,
given each material's supplier lead time and safety-stock policy.
"""

from __future__ import annotations

import pandas as pd


def latest_on_hand(inventory: pd.DataFrame) -> pd.Series:
    """Most recent on-hand snapshot per material (see datagen.facts, which
    logs an `OnHandSnapshot` row after every day's activity)."""
    snapshots = inventory[inventory["transaction_type"] == "OnHandSnapshot"].copy()
    snapshots["date"] = pd.to_datetime(snapshots["date"])
    latest = snapshots.sort_values("date").groupby("material_id").tail(1)
    return latest.set_index("material_id")["quantity"]


def explode_requirements(production_plan: pd.DataFrame, bom: pd.DataFrame) -> pd.DataFrame:
    """Weekly material requirement = sum over products of allocated_units * qty_per_unit."""
    merged = production_plan.merge(bom, on="product_id", how="left")
    merged["required_quantity"] = merged["allocated_units"] * merged["qty_per_unit"]
    requirements = merged.groupby(["week_start", "material_id"], as_index=False)[
        "required_quantity"
    ].sum()
    return requirements.sort_values(["material_id", "week_start"])


def net_requirements_and_recommend_orders(
    requirements: pd.DataFrame, raw_materials: pd.DataFrame, starting_on_hand: pd.Series
) -> pd.DataFrame:
    """Walk each material forward week by week, netting requirements against
    a running projected on-hand balance, and recommend a purchase order
    whenever projected stock would fall below the safety-stock target
    before an order placed today could arrive.

    Recommended orders are assumed placed and received on time — they're
    scheduled as a future delivery `lead_weeks` out and credited back to
    `on_hand` when that week arrives — so the projection reflects a
    functioning reorder policy rather than a purely depleting balance with
    no replenishment. At most one order is kept open per material at a
    time, mirroring a simple "don't double-order" purchasing policy.
    """
    rows = []
    materials = raw_materials.set_index("material_id")

    for material_id, group in requirements.groupby("material_id"):
        group = group.reset_index(drop=True).sort_values("week_start").reset_index(drop=True)
        n_weeks = len(group)
        lead_weeks = max(round(materials.loc[material_id, "lead_time_days"] / 7), 1)
        safety_stock = (
            materials.loc[material_id, "safety_stock_days"] / 7 * group["required_quantity"].mean()
        )

        on_hand = float(starting_on_hand.get(material_id, 0.0))
        pending_deliveries: dict[int, float] = {}  # week index -> quantity arriving that week
        order_open_until: int = -1  # week index; -1 means no open order

        for i, r in group.iterrows():
            delivered_this_week = pending_deliveries.pop(i, 0.0)
            on_hand += delivered_this_week

            future_need = group["required_quantity"].iloc[i : i + lead_weeks].sum()
            will_breach_safety_stock = (on_hand - future_need) < safety_stock

            recommended_order_qty = 0.0
            has_open_order = order_open_until >= i
            if will_breach_safety_stock and not has_open_order:
                recommended_order_qty = max(future_need + safety_stock - on_hand, 0.0)
                arrival_week = i + lead_weeks
                if arrival_week < n_weeks:
                    pending_deliveries[arrival_week] = (
                        pending_deliveries.get(arrival_week, 0.0) + recommended_order_qty
                    )
                    order_open_until = arrival_week
                # if the order would arrive after the planning horizon ends,
                # it's still recommended (so procurement can act on it) but
                # has nothing left to net against within this run.

            projected_end_of_week = on_hand - r["required_quantity"]

            rows.append(
                {
                    "week_start": r["week_start"],
                    "material_id": material_id,
                    "required_quantity": round(r["required_quantity"], 2),
                    "projected_on_hand_start": round(on_hand, 2),
                    "projected_on_hand_end": round(projected_end_of_week, 2),
                    "below_safety_stock": bool(on_hand < safety_stock),
                    "recommended_order_qty": round(recommended_order_qty, 2),
                }
            )
            on_hand = projected_end_of_week

    return pd.DataFrame(rows)
