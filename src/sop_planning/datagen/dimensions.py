"""Generate the dimension tables for the synthetic food manufacturing dataset.

Everything here is fictional: a made-up company ("Zarrin Cake & Confectionery Industries Co."),
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

# (product_name, category, primary_line_id, unit_weight_g, shelf_life_days, unit_price_irr,
#  is_seasonal, seasonal_peak)
PRODUCTS = [
    ("Classic Vanilla Sponge Cake", "Sponge Cake", "L1", 400, 21, 185000, False, None),
    ("Chocolate Sponge Cake", "Sponge Cake", "L1", 400, 21, 195000, False, None),
    ("Lemon Yogurt Sponge Cake", "Sponge Cake", "L1", 380, 18, 175000, False, None),
    ("Honey Sponge Roll", "Sponge Cake", "L1", 300, 14, 165000, False, None),
    ("Vanilla Cream Cake", "Cream Cake", "L2", 450, 10, 245000, False, None),
    ("Chocolate Cream Cake", "Cream Cake", "L2", 450, 10, 255000, False, None),
    ("Strawberry Cream Roll", "Cream Cake", "L2", 350, 8, 235000, False, None),
    ("Hazelnut Cream Cake", "Cream Cake", "L2", 420, 9, 265000, False, None),
    ("Walnut Kolucheh", "Kolucheh", "L3", 55, 45, 68000, True, "Nowruz"),
    ("Date Kolucheh", "Kolucheh", "L3", 60, 45, 65000, True, "Nowruz"),
    ("Cardamom Kolucheh", "Kolucheh", "L3", 50, 45, 62000, True, "Nowruz"),
    ("Fars-style Kolucheh (Rice Flour)", "Kolucheh", "L3", 55, 40, 70000, True, "Nowruz"),
    ("Plain Butter Kolucheh", "Kolucheh", "L3", 50, 45, 58000, False, None),
    ("Rosewater Kolucheh", "Kolucheh", "L3", 55, 45, 66000, True, "Yalda"),
    ("Vanilla Wafer", "Wafer", "L4", 45, 90, 32000, False, None),
    ("Chocolate Wafer", "Wafer", "L4", 45, 90, 34000, False, None),
    ("Hazelnut Wafer", "Wafer", "L4", 45, 90, 36000, False, None),
    ("Plain Butter Biscuit", "Cookie", "L4", 200, 120, 48000, False, None),
    ("Oat & Honey Cookie", "Cookie", "L4", 200, 100, 55000, False, None),
    ("Cocoa Sandwich Cookie", "Cookie", "L4", 180, 100, 52000, False, None),
    ("Yalda Night Gift Box", "Seasonal", "L5", 900, 30, 480000, True, "Yalda"),
    ("Nowruz Sweets Assortment Box", "Seasonal", "L5", 1000, 40, 520000, True, "Nowruz"),
    ("Ramadan Date & Nut Pack", "Seasonal", "L5", 500, 60, 310000, True, "Ramadan"),
    ("Summer Fruit Roll Cake", "Seasonal", "L5", 380, 12, 210000, True, "Summer"),
    ("Back-to-School Snack Pack", "Seasonal", "L5", 250, 60, 145000, True, "AutumnStart"),
]

SUPPLIERS = [
    # (supplier_id, supplier_name, material_categories, avg_lead_time_days,
    #  reliability_score, payment_terms_days)
    # Payment terms follow real purchasing leverage: perishable dairy is
    # near cash-on-delivery (small local suppliers, no float to extend),
    # while dry goods, packaging, and imported flavorings carry standard
    # net-30/45 commercial terms. This is the supplier side of the cash
    # conversion cycle in analytics/working_capital.py.
    ("SUP01", "Alborz Flour Mills", "Wheat Flour, Rice Flour", 5, 0.97, 40),
    ("SUP02", "Karaj Sugar Trading Co.", "Sugar", 4, 0.98, 35),
    ("SUP03", "Fresh Dairy Distributors", "Eggs, Butter, Milk Powder", 2, 0.95, 10),
    ("SUP04", "Golestan Nut & Dried Fruit Co.", "Walnuts, Dates, Almonds", 7, 0.93, 30),
    (
        "SUP05",
        "Persia Cocoa & Flavoring Imports",
        "Cocoa Powder, Vanilla, Cardamom, Rosewater",
        12,
        0.90,
        40,
    ),
    ("SUP06", "National Packaging Industries", "Packaging Film, Cartons, Labels", 6, 0.96, 45),
    ("SUP07", "Alborz Vegetable Oil Refinery", "Vegetable Oil, Baking Powder, Yeast", 5, 0.97, 30),
    ("SUP08", "Sepahan Industrial Sugar & Additives", "Sugar, Baking Powder", 5, 0.94, 35),
]

# (material_name, unit, unit_cost_irr, supplier_id, lead_time_days, safety_stock_days)
RAW_MATERIALS = [
    ("Wheat Flour", "kg", 28000, "SUP01", 5, 7),
    ("Rice Flour", "kg", 42000, "SUP01", 6, 10),
    ("Sugar", "kg", 45000, "SUP02", 4, 7),
    ("Eggs", "unit", 6500, "SUP03", 2, 3),
    ("Butter", "kg", 210000, "SUP03", 3, 5),
    ("Milk Powder", "kg", 165000, "SUP03", 4, 7),
    ("Vegetable Oil", "L", 95000, "SUP07", 5, 7),
    ("Cocoa Powder", "kg", 320000, "SUP05", 12, 15),
    ("Walnuts", "kg", 480000, "SUP04", 7, 10),
    ("Dates", "kg", 190000, "SUP04", 7, 10),
    ("Cardamom", "kg", 950000, "SUP05", 12, 20),
    ("Rosewater", "L", 140000, "SUP05", 12, 15),
    ("Vanilla Extract", "L", 380000, "SUP05", 12, 15),
    ("Baking Powder", "kg", 65000, "SUP07", 5, 10),
    ("Yeast", "kg", 120000, "SUP07", 4, 7),
    ("Packaging Film", "unit", 1800, "SUP06", 6, 10),
    ("Cartons", "unit", 4200, "SUP06", 6, 10),
    ("Gift Box Packaging", "unit", 22000, "SUP06", 8, 12),
]

REGIONS = [
    # (region_id, region_name, customer_type, payment_terms_days)
    # Payment terms reflect real buying power: large retail chains pay
    # fastest, smaller wholesale distributors slower, export slowest of all
    # (letters of credit + customs). This asymmetry with supplier payment
    # terms (see SUPPLIERS below) is the main driver of the cash conversion
    # cycle computed in analytics/working_capital.py.
    ("R01", "Tehran Metro", "Retail Chain", 50),
    ("R02", "Alborz (Karaj)", "Wholesale Distributor", 80),
    ("R03", "Isfahan", "Retail Chain", 55),
    ("R04", "Fars (Shiraz)", "Wholesale Distributor", 85),
    ("R05", "Khorasan Razavi (Mashhad)", "Retail Chain", 60),
    ("R06", "East Azerbaijan (Tabriz)", "Wholesale Distributor", 90),
    ("R07", "National Export", "Export", 120),
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
        monthly_wage_irr = int(rng.normal(base_by_role[role] * 55_000_000, 4_000_000))

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
                "monthly_wage_irr": max(monthly_wage_irr, 40_000_000),
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
        "unit_price_irr",
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
        "unit_cost_irr",
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
