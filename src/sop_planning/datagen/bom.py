"""Bill of Materials (BOM): how much of each raw material one unit of each
product consumes. Recipes are approximate but plausible per product category
so the downstream MRP explosion produces sane material requirements.
"""

from __future__ import annotations

import pandas as pd

# Category -> {material_name: qty per unit produced}, in the material's unit
# (kg, L, or unit — see dimensions.RAW_MATERIALS).
CATEGORY_RECIPES = {
    "Sponge Cake": {
        "Wheat Flour": 0.12,
        "Sugar": 0.09,
        "Eggs": 0.35,
        "Butter": 0.05,
        "Vegetable Oil": 0.02,
        "Baking Powder": 0.004,
        "Vanilla Extract": 0.002,
        "Packaging Film": 1.0,
        "Cartons": 0.25,
    },
    "Cream Cake": {
        "Wheat Flour": 0.10,
        "Sugar": 0.11,
        "Eggs": 0.30,
        "Butter": 0.09,
        "Milk Powder": 0.03,
        "Cocoa Powder": 0.02,
        "Baking Powder": 0.003,
        "Vanilla Extract": 0.003,
        "Packaging Film": 1.0,
        "Cartons": 0.25,
    },
    "Kolucheh": {
        "Wheat Flour": 0.035,
        "Rice Flour": 0.010,
        "Sugar": 0.012,
        "Butter": 0.015,
        "Walnuts": 0.008,
        "Dates": 0.006,
        "Cardamom": 0.0004,
        "Rosewater": 0.0003,
        "Yeast": 0.001,
        "Packaging Film": 1.0,
    },
    "Wafer": {
        "Wheat Flour": 0.018,
        "Sugar": 0.010,
        "Vegetable Oil": 0.008,
        "Cocoa Powder": 0.004,
        "Baking Powder": 0.0008,
        "Packaging Film": 1.0,
    },
    "Cookie": {
        "Wheat Flour": 0.09,
        "Sugar": 0.05,
        "Butter": 0.04,
        "Eggs": 0.02,
        "Cocoa Powder": 0.015,
        "Baking Powder": 0.002,
        "Packaging Film": 1.0,
        "Cartons": 0.2,
    },
    "Seasonal": {
        "Wheat Flour": 0.05,
        "Sugar": 0.04,
        "Butter": 0.03,
        "Walnuts": 0.02,
        "Dates": 0.03,
        "Cardamom": 0.001,
        "Rosewater": 0.0005,
        "Gift Box Packaging": 1.0,
    },
}


def generate_bom(products: pd.DataFrame, raw_materials: pd.DataFrame) -> pd.DataFrame:
    material_id_by_name = dict(zip(raw_materials["material_name"], raw_materials["material_id"]))

    rows = []
    for _, product in products.iterrows():
        recipe = CATEGORY_RECIPES[product["category"]]
        for material_name, qty_per_unit in recipe.items():
            rows.append(
                {
                    "product_id": product["product_id"],
                    "material_id": material_id_by_name[material_name],
                    "qty_per_unit": qty_per_unit,
                }
            )
    return pd.DataFrame(rows)
