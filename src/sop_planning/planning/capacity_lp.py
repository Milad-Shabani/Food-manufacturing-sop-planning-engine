"""Production capacity planning as a linear program.

Each production line is shared across several products, and both machine
time and labor are finite. When forecasted demand for a week exceeds what
a line can actually produce, *something* has to give — the question this
module answers is: which products should be prioritized so the plant
captures as much revenue as possible, given both constraints?

For each (line, week):

    maximize      sum_p  unit_price[p] * units[p]
    subject to    0 <= units[p] <= forecast[p]                    for every product p on the line
                  sum_p  units[p] / capacity_units_per_hour   <= machine_hours_available
                  sum_p  units[p] * labor_hours_per_unit[p]   <= labor_hours_available

`labor_hours_available` comes from the actual simulated workforce
attendance for that line/week (not the theoretical headcount), so a
short-staffed week genuinely tightens the plan. `labor_hours_per_unit` is
derived from a per-line "operators needed to run one hour at full
capacity" assumption in `config.yaml` — a documented planning assumption,
not something inferred from the HR data.

This has a closed-form greedy solution (a fractional multi-constraint
knapsack), but it's solved with `scipy.optimize.linprog` so the model can
grow additional constraints later without a rewrite.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


def build_labor_hours_per_unit(
    products: pd.DataFrame, lines: pd.DataFrame, operators_per_running_hour: dict[str, float]
) -> dict[str, float]:
    """labor_hours_per_unit[product] = operators_needed_per_hour / units_per_hour."""
    capacity_by_line = dict(zip(lines["line_id"], lines["capacity_units_per_hour"]))
    result = {}
    for _, p in products.iterrows():
        line_id = p["primary_line_id"]
        result[p["product_id"]] = operators_per_running_hour[line_id] / capacity_by_line[line_id]
    return result


def plan_line_week(
    products_on_line: pd.DataFrame,
    forecast_by_product: dict[str, float],
    capacity_units_per_hour: float,
    machine_hours_available: float,
    labor_hours_available: float,
    labor_hours_per_unit: dict[str, float],
) -> pd.DataFrame:
    """Solve the LP for one line, one week. Returns allocated units per product."""
    product_ids = products_on_line["product_id"].tolist()
    n = len(product_ids)
    if n == 0:
        return pd.DataFrame(columns=["product_id", "allocated_units", "forecast_units"])

    prices = products_on_line.set_index("product_id")["unit_price_irr"].reindex(product_ids).values
    forecasts = np.array([forecast_by_product.get(pid, 0.0) for pid in product_ids])

    # linprog minimizes, so negate the revenue objective to maximize it.
    c = -prices

    machine_row = np.array([1.0 / capacity_units_per_hour] * n)
    labor_row = np.array([labor_hours_per_unit[pid] for pid in product_ids])
    A_ub = np.vstack([machine_row, labor_row])
    b_ub = np.array([machine_hours_available, labor_hours_available])

    bounds = [(0, f) for f in forecasts]

    if forecasts.sum() == 0:
        allocated = np.zeros(n)
    else:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success:
            logger.warning(
                "LP did not converge for line week (products=%s); falling back to 0 allocation.",
                product_ids,
            )
            allocated = np.zeros(n)
        else:
            allocated = np.clip(np.round(result.x), 0, forecasts)

    return pd.DataFrame(
        {
            "product_id": product_ids,
            "allocated_units": allocated.astype(int),
            "forecast_units": forecasts.astype(int),
        }
    )


def build_production_plan(
    forecast: pd.DataFrame,
    products: pd.DataFrame,
    lines: pd.DataFrame,
    workforce: pd.DataFrame,
    operators_per_running_hour: dict[str, float],
    hours_per_shift: float = 8.0,
    oee: float = 1.0,
) -> pd.DataFrame:
    """Run the per-line-per-week LP across the whole forecast horizon."""
    labor_hours_per_unit = build_labor_hours_per_unit(products, lines, operators_per_running_hour)
    capacity_by_line = dict(zip(lines["line_id"], lines["capacity_units_per_hour"]))
    shifts_by_line = dict(zip(lines["line_id"], lines["shifts_per_day"]))

    workforce = workforce.copy()
    workforce["date"] = pd.to_datetime(workforce["date"])
    workforce["week_start"] = workforce["date"].dt.to_period("W-SAT").apply(lambda p: p.start_time)
    weekly_labor_hours = (
        workforce.groupby(["line_id", "week_start"])["present_headcount"].sum() * hours_per_shift
    )
    # The forecast horizon extends beyond the historical workforce log (we
    # don't have a crystal ball for future attendance), so weeks outside
    # that history fall back to each line's own historical average weekly
    # labor-hours — grounded in that line's actual attendance pattern
    # rather than a generic assumption.
    avg_labor_hours_by_line = weekly_labor_hours.groupby(level=0).mean()

    plans = []
    for week_start in sorted(forecast["week_start"].unique()):
        week_forecast = forecast[forecast["week_start"] == week_start]
        forecast_by_product = dict(
            zip(week_forecast["product_id"], week_forecast["forecast_units"])
        )

        for line_id in lines["line_id"]:
            line_products = products[products["primary_line_id"] == line_id]
            machine_hours = shifts_by_line[line_id] * hours_per_shift * 7 * oee
            labor_hours = weekly_labor_hours.get(
                (line_id, week_start), avg_labor_hours_by_line.get(line_id, machine_hours)
            )

            plan = plan_line_week(
                line_products,
                forecast_by_product,
                capacity_units_per_hour=capacity_by_line[line_id],
                machine_hours_available=machine_hours,
                labor_hours_available=labor_hours,
                labor_hours_per_unit=labor_hours_per_unit,
            )
            plan["line_id"] = line_id
            plan["week_start"] = week_start
            plans.append(plan)

    result = pd.concat(plans, ignore_index=True) if plans else pd.DataFrame()
    if not result.empty:
        result["unmet_units"] = result["forecast_units"] - result["allocated_units"]
    return result
