"""Persist planning outputs: a local SQLite warehouse (for ad hoc SQL /
Power BI) and Parquet (for anything else). Kept intentionally simple —
this module's job is publishing results, not re-solving the same
multi-backend-upsert problem already covered in the sibling ETL project.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

TABLE_NAMES = {
    "forecast": "fact_demand_forecast",
    "production_plan": "fact_production_plan",
    "material_requirements": "fact_material_requirements",
    "weekly_summary": "fact_sop_weekly_summary",
    "backtest_accuracy": "fact_forecast_accuracy",
    "working_capital": "fact_working_capital",
}


def write_outputs(tables: dict[str, pd.DataFrame], processed_dir: str, parquet_dir: str) -> None:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    db_path = processed_path / "planning_warehouse.db"

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    parquet_path = Path(parquet_dir)
    parquet_path.mkdir(parents=True, exist_ok=True)

    for key, df in tables.items():
        table_name = TABLE_NAMES[key]
        out_df = df.copy()
        for col in out_df.columns:
            if "week_start" in col or col == "date":
                out_df[col] = out_df[col].astype(str)

        out_df.to_sql(table_name, engine, if_exists="replace", index=False)
        out_df.to_parquet(parquet_path / f"{table_name}.parquet", index=False)
        logger.info("Wrote %s (%d rows) to warehouse + Parquet", table_name, len(out_df))
