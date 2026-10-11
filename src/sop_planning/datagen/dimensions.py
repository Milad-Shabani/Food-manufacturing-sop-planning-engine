"""Generate the dimension tables for the synthetic food manufacturing dataset.

Everything here is fictional: a made-up cake & confectionery factory,
made-up employees, made-up suppliers. Values are randomized but seeded, so
the same seed always reproduces the same dataset — useful for a portfolio
repo where reviewers should see the same numbers you describe in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

ROLES = [
    ("Production Operator", 0.42),
    ("Line Supervisor", 0.04),
    ("QC Inspector", 0.06),
    ("Warehouse Staff", 0.10),
    ("Forklift Operator", 0.04),
    ("Maintenance Technician", 0.05),
    ("Logistics Coordinator", 0.05),
    ("Sales Representative", 0.10),
    ("Procurement Officer", 0.03),
    ("Finance / Admin", 0.06),
    ("Plant Management", 0.02),
    ("HR / Support", 0.03),
]

SHIFTS = ["Morning (06:00-14:00)", "Evening (14:00-22:00)", "Night (22:00-06:00)"]

PRODUCTION_LINES = [
    {
        "line_id": "L1",
        "line_name": "Sponge Cake Line",
        "capacity_units_per_hour": 900,
        "shifts_per_day": 2,
        "changeover_minutes": 25,
        "commissioned_date": "2016-03-01",
    },
    {
        "line_id": "L2",
        "line_name": "Cream-Filled Cake Line",
        "capacity_units_per_hour": 650,
        "shifts_per_day": 2,
        "changeover_minutes": 35,
        "commissioned_date": "2017-08-15",
    },
    {
        "line_id": "L3",
        "line_name": "Traditional Kolucheh Line",
        "capacity_units_per_hour": 220,
        "shifts_per_day": 3,
        "changeover_minutes": 20,
        "commissioned_date": "2015-01-10",
    },
    {
        "line_id": "L4",
        "line_name": "Wafer & Biscuit Line",
        "capacity_units_per_hour": 1400,
        "shifts_per_day": 2,
        "changeover_minutes": 15,
        "commissioned_date": "2019-05-20",
    },
    {
        "line_id": "L5",
        "line_name": "Seasonal & Holiday Line",
        "capacity_units_per_hour": 750,
        "shifts_per_day": 2,
        "changeover_minutes": 30,
        "commissioned_date": "2020-11-01",
    },
]

# (product_name, category, primary_line_id, unit_weight_g, shelf_life_days, unit_price_usd,
#  is_seasonal, seasonal_peak)
# Kolucheh is sold as a pack of 6 pieces (price and weight are per pack).
PRODUCTS = [
    ("Classic Vanilla Sponge Cake", "Sponge Cake", "L1", 400, 21, 1.85, False, None),
    ("Chocolate Sponge Cake", "Sponge Cake", "L1", 400, 21, 1.95, False, None),
    ("Lemon Yogurt Sponge Cake", "Sponge Cake", "L1", 380, 18, 1.75, False, None),
    ("Honey Sponge Roll", "Sponge Cake", "L1", 300, 14, 1.65, False, None),
    ("Vanilla Cream Cake", "Cream Cake", "L2", 450, 10, 2.45, False, None),
    ("Chocolate Cream Cake", "Cream Cake", "L2", 450, 10, 2.55, False, None),
    ("Strawberry Cream Roll", "Cream Cake", "L2", 350, 8, 2.35, False, None),
    ("Hazelnut Cream Cake", "Cream Cake", "L2", 420, 9, 2.65, False, None),
    ("Walnut Kolucheh", "Kolucheh", "L3", 330, 45, 4.08, True, "Nowruz"),
    ("Date Kolucheh", "Kolucheh", "L3", 360, 45, 3.90, True, "Nowruz"),
    ("Cardamom Kolucheh", "Kolucheh", "L3", 300, 45, 3.72, True, "Nowruz"),
    ("Fars-style Kolucheh (Rice Flour)", "Kolucheh", "L3", 330, 40, 4.20, True, "Nowruz"),
    ("Plain Butter Kolucheh", "Kolucheh", "L3", 300, 45, 3.48, False, None),
    ("Rosewater Kolucheh", "Kolucheh", "L3", 330, 45, 3.96, True, "Yalda"),
    ("Vanilla Wafer", "Wafer", "L4", 45, 90, 0.32, False, None),
    ("Chocolate Wafer", "Wafer", "L4", 45, 90, 0.34, False, None),
    ("Hazelnut Wafer", "Wafer", "L4", 45, 90, 0.36, False, None),
    ("Plain Butter Biscuit", "Cookie", "L4", 200, 120, 1.15, False, None),
    ("Oat & Honey Cookie", "Cookie", "L4", 200, 100, 1.30, False, None),
    ("Cocoa Sandwich Cookie", "Cookie", "L4", 180, 100, 1.25, False, None),
    ("Yalda Night Gift Box", "Seasonal", "L5", 900, 30, 4.80, True, "Yalda"),
    ("Nowruz Sweets Assortment Box", "Seasonal", "L5", 1000, 40, 5.20, True, "Nowruz"),
    ("Ramadan Date & Nut Pack", "Seasonal", "L5", 500, 60, 3.10, True, "Ramadan"),
    ("Summer Fruit Roll Cake", "Seasonal", "L5", 380, 12, 3.40, True, "Summer"),
    ("Back-to-School Snack Pack", "Seasonal", "L5", 250, 60, 2.60, True, "AutumnStart"),
]

SUPPLIERS = [
    # (supplier_id, supplier_name, material_categories, avg_lead_time_days,
    #  reliability_score, payment_terms_days)
    # Payment terms follow real purchasing leverage: perishable dairy is
    # near cash-on-delivery (small local suppliers, no float to extend),
    # while dry goods, packaging, and imported flavorings carry standard
    # net-30/45 commercial terms. This is the supplier side of the cash
    # conversion cycle in analytics/working_capital.py.
    ("SUP01", "Supplier 1", "Wheat Flour, Rice Flour", 5, 0.97, 40),
    ("SUP02", "Supplier 2", "Sugar", 4, 0.98, 35),
    ("SUP03", "Supplier 3", "Eggs, Butter, Milk Powder", 2, 0.95, 10),
    ("SUP04", "Supplier 4", "Walnuts, Dates, Almonds", 7, 0.93, 30),
    (
        "SUP05",
        "Supplier 5",
        "Cocoa Powder, Vanilla, Cardamom, Rosewater",
        12,
        0.90,
        40,
    ),
    ("SUP06", "Supplier 6", "Packaging Film, Cartons, Labels", 6, 0.96, 45),
    ("SUP07", "Supplier 7", "Vegetable Oil, Baking Powder, Yeast", 5, 0.97, 30),
    ("SUP08", "Supplier 8", "Sugar, Baking Powder", 5, 0.94, 35),
]

# (material_name, unit, unit_cost_usd, supplier_id, lead_time_days, safety_stock_days)
RAW_MATERIALS = [
    ("Wheat Flour", "kg", 0.76, "SUP01", 5, 7),
    ("Rice Flour", "kg", 1.13, "SUP01", 6, 10),
    ("Sugar", "kg", 1.21, "SUP02", 4, 7),
    ("Eggs", "unit", 0.18, "SUP03", 2, 3),
    ("Butter", "kg", 5.67, "SUP03", 3, 5),
    ("Milk Powder", "kg", 4.46, "SUP03", 4, 7),
    ("Vegetable Oil", "L", 2.56, "SUP07", 5, 7),
    ("Cocoa Powder", "kg", 8.64, "SUP05", 12, 15),
    ("Walnuts", "kg", 12.96, "SUP04", 7, 10),
    ("Dates", "kg", 5.13, "SUP04", 7, 10),
    ("Cardamom", "kg", 25.65, "SUP05", 12, 20),
    ("Rosewater", "L", 3.78, "SUP05", 12, 15),
    ("Vanilla Extract", "L", 10.26, "SUP05", 12, 15),
    ("Baking Powder", "kg", 1.75, "SUP07", 5, 10),
    ("Yeast", "kg", 3.24, "SUP07", 4, 7),
    ("Packaging Film", "unit", 0.05, "SUP06", 6, 10),
    ("Cartons", "unit", 0.11, "SUP06", 6, 10),
    ("Gift Box Packaging", "unit", 0.59, "SUP06", 8, 12),
]

REGIONS = [
    # (region_id, region_name, customer_type, payment_terms_days)
    # Payment terms reflect real buying power: large retail chains pay
    # fastest, smaller wholesale distributors slower, export slowest of all
    # (letters of credit + customs). This asymmetry with supplier payment
    # terms (see SUPPLIERS below) is the main driver of the cash conversion
    # cycle computed in analytics/working_capital.py.
    ("R01", "Region 1", "Retail Chain", 50),
    ("R02", "Region 2", "Wholesale Distributor", 80),
    ("R03", "Region 3", "Retail Chain", 55),
    ("R04", "Region 4", "Wholesale Distributor", 85),
    ("R05", "Region 5", "Retail Chain", 60),
    ("R06", "Region 6", "Wholesale Distributor", 90),
    ("R07", "Export", "Export", 120),
]


def generate_employees(engine_seed: int, n_employees: int = 200) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(engine_seed)
    rng = np.random.default_rng(engine_seed)

    role_names = [r[0] for r in ROLES]
    role_weights = np.array([r[1] for r in ROLES])
    role_weights = role_weights / role_weights.sum()

    lines = [p["line_id"] for p in PRODUCTION_LINES]
    rows = []
    for i in range(1, n_employees + 1):
        role = rng.choice(role_names, p=role_weights)
        is_production = role in ("Production Operator", "Line Supervisor", "QC Inspector")
        line_id = rng.choice(lines) if is_production else None
        shift = (
            rng.choice(SHIFTS)
            if role not in ("Plant Management", "Finance / Admin", "HR / Support")
            else "Day Office"
        )
        hire_date = fake.date_between(start_date="-9y", end_date="-30d")
        employment_type = rng.choice(
            ["Full-time", "Full-time", "Full-time", "Part-time", "Seasonal"],
            p=[0.55, 0.15, 0.1, 0.1, 0.1],
        )
        base_by_role = {
            "Production Operator": 1.0,
            "Line Supervisor": 1.6,
            "QC Inspector": 1.3,
            "Warehouse Staff": 0.95,
            "Forklift Operator": 1.05,
            "Maintenance Technician": 1.4,
            "Logistics Coordinator": 1.25,
            "Sales Representative": 1.3,
            "Procurement Officer": 1.35,
            "Finance / Admin": 1.3,
            "Plant Management": 2.4,
            "HR / Support": 1.1,
        }
        monthly_wage_usd = int(rng.normal(base_by_role[role] * 1_250, 100))

        rows.append(
            {
                "employee_id": f"E{i:04d}",
                "full_name": fake.name(),
                "role": role,
                "department": (
                    "Production"
                    if is_production
                    else (
                        "Warehouse & Logistics"
                        if role in ("Warehouse Staff", "Forklift Operator", "Logistics Coordinator")
                        else "Corporate"
                    )
                ),
                "production_line_id": line_id,
                "shift": shift,
                "employment_type": employment_type,
                "hire_date": hire_date,
                "monthly_wage_usd": max(monthly_wage_usd, 950),
            }
        )
    return pd.DataFrame(rows)


def generate_production_lines() -> pd.DataFrame:
    return pd.DataFrame(PRODUCTION_LINES)


def generate_products() -> pd.DataFrame:
    cols = [
        "product_name",
        "category",
        "primary_line_id",
        "unit_weight_g",
        "shelf_life_days",
        "unit_price_usd",
        "is_seasonal",
        "seasonal_peak",
    ]
    df = pd.DataFrame(PRODUCTS, columns=cols)
    df.insert(0, "product_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])
    return df


def generate_suppliers() -> pd.DataFrame:
    cols = [
        "supplier_id",
        "supplier_name",
        "material_categories",
        "avg_lead_time_days",
        "reliability_score",
        "payment_terms_days",
    ]
    return pd.DataFrame(SUPPLIERS, columns=cols)


def generate_raw_materials() -> pd.DataFrame:
    cols = [
        "material_name",
        "unit_of_measure",
        "unit_cost_usd",
        "default_supplier_id",
        "lead_time_days",
        "safety_stock_days",
    ]
    df = pd.DataFrame(RAW_MATERIALS, columns=cols)
    df.insert(0, "material_id", [f"M{i:02d}" for i in range(1, len(df) + 1)])
    return df


def generate_regions() -> pd.DataFrame:
    cols = ["region_id", "region_name", "customer_type", "payment_terms_days"]
    return pd.DataFrame(REGIONS, columns=cols)
