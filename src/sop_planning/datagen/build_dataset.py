"""Entry point that generates the full synthetic dataset and writes it to
`data/raw/` as CSVs.

    python -m sop_planning.datagen.build_dataset --out-dir data/raw --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import dimensions, facts
from .bom import generate_bom

logger = logging.getLogger(__name__)


def build_dataset(
    out_dir: str,
    seed: int = 42,
    n_employees: int = 200,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
) -> dict[str, int]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating dimension tables...")
    employees = dimensions.generate_employees(seed, n_employees=n_employees)
    lines = dimensions.generate_production_lines()
    products = dimensions.generate_products()
    suppliers = dimensions.generate_suppliers()
    raw_materials = dimensions.generate_raw_materials()
    regions = dimensions.generate_regions()
    bom = generate_bom(products, raw_materials)

    dates = facts.generate_date_range(start, end)

    logger.info("Generating %d days of sales history for %d products...", len(dates), len(products))
    sales = facts.generate_sales_history(products, regions, dates, seed=seed)

    logger.info("Deriving production log from sales demand...")
    production = facts.generate_production_log(sales, products, lines, seed=seed + 1)

    logger.info("Exploding production through the BOM into material consumption...")
    consumption = facts.generate_material_consumption(production, bom)

    logger.info("Simulating purchasing policy and inventory ledger...")
    purchase_orders, inventory = facts.generate_purchase_orders_and_inventory(
        consumption, raw_materials, dates, seed=seed + 2
    )

    logger.info("Simulating daily workforce availability...")
    workforce = facts.generate_workforce_availability(employees, lines, dates, seed=seed + 3)

    tables = {
        "dim_employee.csv": employees,
        "dim_production_line.csv": lines,
        "dim_product.csv": products,
        "dim_supplier.csv": suppliers,
        "dim_raw_material.csv": raw_materials,
        "dim_region.csv": regions,
        "bom.csv": bom,
        "fact_sales_history.csv": sales,
        "fact_production_log.csv": production,
        "fact_inventory_transactions.csv": inventory,
        "fact_purchase_orders.csv": purchase_orders,
        "fact_workforce_availability.csv": workforce,
    }

    row_counts = {}
    for filename, df in tables.items():
        df.to_csv(out_path / filename, index=False)
        row_counts[filename] = len(df)
        logger.info("Wrote %s (%d rows)", filename, len(df))

    return row_counts


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--employees", type=int, default=200)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    counts = build_dataset(
        args.out_dir, seed=args.seed, n_employees=args.employees, start=args.start, end=args.end
    )
    total = sum(counts.values())
    print(f"\nGenerated {len(counts)} tables, {total:,} total rows, in {args.out_dir}")


if __name__ == "__main__":
    main()
