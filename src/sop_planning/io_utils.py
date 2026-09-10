"""Load the raw CSV tables (produced by `datagen.build_dataset`) into memory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_tables(raw_dir: str) -> dict[str, pd.DataFrame]:
    raw_path = Path(raw_dir)
    names = [
        "dim_employee",
        "dim_production_line",
        "dim_product",
        "dim_supplier",
        "dim_raw_material",
        "dim_region",
        "bom",
        "fact_sales_history",
        "fact_production_log",
        "fact_inventory_transactions",
        "fact_purchase_orders",
        "fact_workforce_availability",
    ]
    tables = {}
    for name in names:
        path = raw_path / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python -m sop_planning.datagen.build_dataset` first."
            )
        tables[name] = pd.read_csv(path)
    return tables
