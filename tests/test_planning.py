import pandas as pd

from sop_planning.planning.capacity_lp import build_labor_hours_per_unit, plan_line_week
from sop_planning.planning.mrp import explode_requirements, net_requirements_and_recommend_orders


def _products_on_line():
    return pd.DataFrame(
        [
            {"product_id": "P1", "unit_price_irr": 100000, "primary_line_id": "L1"},
            {"product_id": "P2", "unit_price_irr": 300000, "primary_line_id": "L1"},
        ]
    )


def test_plan_line_week_respects_capacity_and_never_exceeds_forecast():
    products = _products_on_line()
    forecast = {"P1": 1000, "P2": 1000}
    labor_hours_per_unit = {"P1": 0.01, "P2": 0.01}

    plan = plan_line_week(
        products,
        forecast,
        capacity_units_per_hour=100,  # tight: 100 units/hr * 10 hrs = 1000 total units max
        machine_hours_available=10,
        labor_hours_available=1000,  # not the binding constraint here
        labor_hours_per_unit=labor_hours_per_unit,
    )

    assert (plan["allocated_units"] <= plan["forecast_units"]).all()
    assert plan["allocated_units"].sum() <= 1000 + 1  # rounding slack


def test_plan_line_week_prioritizes_higher_revenue_product_when_capacity_binds():
    products = _products_on_line()  # P2 is 3x the price of P1
    forecast = {"P1": 1000, "P2": 1000}
    labor_hours_per_unit = {"P1": 0.01, "P2": 0.01}

    plan = plan_line_week(
        products,
        forecast,
        capacity_units_per_hour=100,
        machine_hours_available=10,  # only enough for 1000 units total, demand is 2000
        labor_hours_available=1000,
        labor_hours_per_unit=labor_hours_per_unit,
    ).set_index("product_id")

    # The LP should fully satisfy the higher-revenue product (P2) before
    # allocating any remaining capacity to the cheaper one (P1).
    assert plan.loc["P2", "allocated_units"] == plan.loc["P2", "forecast_units"]
    assert plan.loc["P1", "allocated_units"] < plan.loc["P1", "forecast_units"]


def test_plan_line_week_binds_on_labor_even_with_ample_machine_hours():
    products = _products_on_line()
    forecast = {"P1": 1000, "P2": 1000}
    labor_hours_per_unit = {"P1": 0.02, "P2": 0.02}  # 2000 units would need 40 labor-hours

    plan = plan_line_week(
        products,
        forecast,
        capacity_units_per_hour=1000,  # machine hours are not the bottleneck
        machine_hours_available=100,
        labor_hours_available=20,  # only enough for 1000 units total
        labor_hours_per_unit=labor_hours_per_unit,
    )

    assert plan["allocated_units"].sum() <= 1000 + 1


def test_build_labor_hours_per_unit_matches_expected_ratio():
    products = pd.DataFrame([{"product_id": "P1", "primary_line_id": "L1"}])
    lines = pd.DataFrame([{"line_id": "L1", "capacity_units_per_hour": 200}])

    result = build_labor_hours_per_unit(products, lines, operators_per_running_hour={"L1": 10})
    assert result["P1"] == 10 / 200


def test_explode_requirements_sums_across_products_sharing_a_material():
    plan = pd.DataFrame(
        [
            {"product_id": "P1", "week_start": "2026-01-04", "allocated_units": 100},
            {"product_id": "P2", "week_start": "2026-01-04", "allocated_units": 50},
        ]
    )
    bom = pd.DataFrame(
        [
            {"product_id": "P1", "material_id": "M1", "qty_per_unit": 0.1},
            {"product_id": "P2", "material_id": "M1", "qty_per_unit": 0.2},
        ]
    )
    req = explode_requirements(plan, bom)
    assert req["required_quantity"].iloc[0] == 100 * 0.1 + 50 * 0.2


def test_mrp_recommends_an_order_when_stock_would_run_out_and_replenishes_it():
    requirements = pd.DataFrame(
        [
            {
                "week_start": pd.Timestamp("2026-01-04"),
                "material_id": "M1",
                "required_quantity": 100,
            },
            {
                "week_start": pd.Timestamp("2026-01-11"),
                "material_id": "M1",
                "required_quantity": 100,
            },
            {
                "week_start": pd.Timestamp("2026-01-18"),
                "material_id": "M1",
                "required_quantity": 100,
            },
            {
                "week_start": pd.Timestamp("2026-01-25"),
                "material_id": "M1",
                "required_quantity": 100,
            },
        ]
    )
    raw_materials = pd.DataFrame(
        [{"material_id": "M1", "lead_time_days": 7, "safety_stock_days": 7}]
    )
    starting_on_hand = pd.Series({"M1": 150})  # only ~1.5 weeks of stock on hand

    result = net_requirements_and_recommend_orders(requirements, raw_materials, starting_on_hand)

    assert result["recommended_order_qty"].sum() > 0
    # once an order is placed and arrives, on-hand should recover rather
    # than drift further and further negative every subsequent week
    assert result["projected_on_hand_end"].iloc[-1] > result["projected_on_hand_end"].min()
