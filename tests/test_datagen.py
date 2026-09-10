import pandas as pd

from sop_planning.datagen import dimensions, facts
from sop_planning.datagen.bom import generate_bom


def test_generate_employees_count_and_columns():
    employees = dimensions.generate_employees(engine_seed=1, n_employees=50)
    assert len(employees) == 50
    assert {"employee_id", "role", "production_line_id", "shift", "monthly_wage_irr"}.issubset(
        employees.columns
    )
    assert employees["employee_id"].is_unique


def test_products_reference_valid_lines():
    products = dimensions.generate_products()
    lines = dimensions.generate_production_lines()
    assert set(products["primary_line_id"]).issubset(set(lines["line_id"]))


def test_bom_references_valid_products_and_materials():
    products = dimensions.generate_products()
    raw_materials = dimensions.generate_raw_materials()
    bom = generate_bom(products, raw_materials)

    assert set(bom["product_id"]).issubset(set(products["product_id"]))
    assert set(bom["material_id"]).issubset(set(raw_materials["material_id"]))
    assert (bom["qty_per_unit"] > 0).all()


def test_sales_history_shows_a_nowruz_spike_for_seasonal_products():
    products = dimensions.generate_products()
    regions = dimensions.generate_regions()
    dates = facts.generate_date_range("2025-01-01", "2025-12-31")

    sales = facts.generate_sales_history(products, regions, dates, seed=7)
    sales["date"] = pd.to_datetime(sales["date"])

    nowruz_product = products[products["seasonal_peak"] == "Nowruz"].iloc[0]["product_id"]
    series = sales[sales["product_id"] == nowruz_product].groupby("date")["units_sold"].sum()

    peak = series.loc["2025-03-10":"2025-03-20"].mean()
    baseline = series.loc["2025-06-01":"2025-06-30"].mean()
    assert peak > baseline * 2  # a real seasonal bump, not noise


def test_production_log_respects_line_capacity():
    products = dimensions.generate_products()
    regions = dimensions.generate_regions()
    lines = dimensions.generate_production_lines()
    dates = facts.generate_date_range("2025-01-01", "2025-03-31")

    sales = facts.generate_sales_history(products, regions, dates, seed=3)
    production = facts.generate_production_log(sales, products, lines, seed=4)

    capacity = dict(zip(lines["line_id"], lines["capacity_units_per_hour"]))
    shifts = dict(zip(lines["line_id"], lines["shifts_per_day"]))

    for _, row in production.iterrows():
        max_possible = capacity[row["line_id"]] * shifts[row["line_id"]] * 8
        assert row["planned_units"] <= max_possible
