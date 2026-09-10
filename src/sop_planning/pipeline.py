"""Pipeline orchestration: load data -> forecast -> backtest -> optimize
production plan -> MRP -> quality gate -> publish.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .analytics.working_capital import compute_ccc_timeseries
from .config import PlanningSettings, load_settings
from .forecasting.backtest import accuracy_summary, backtest
from .forecasting.forecaster import forecast_all_products, weekly_demand
from .io_utils import load_raw_tables
from .load.outputs import write_outputs
from .planning.capacity_lp import build_production_plan
from .planning.mrp import (
    explode_requirements,
    latest_on_hand,
    net_requirements_and_recommend_orders,
)
from .planning.sop_report import build_weekly_summary
from .quality.checks import QualityReport, run_quality_checks

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    weeks_forecasted: int
    total_forecast_units: int
    total_allocated_units: int
    avg_ccc_days: float
    quality: QualityReport
    duration_seconds: float


def run_pipeline(
    config_path: str = "config/config.yaml", fail_on_quality: bool = True
) -> PipelineResult:
    started = time.monotonic()
    settings: PlanningSettings = load_settings(config_path)

    logger.info("Loading raw tables from %s", settings.raw_dir)
    tables = load_raw_tables(settings.raw_dir)

    logger.info("Aggregating daily sales to weekly demand...")
    weekly = weekly_demand(tables["fact_sales_history"])

    logger.info("Backtesting forecast accuracy over the last %d weeks...", settings.backtest_weeks)
    bt = backtest(weekly, test_weeks=settings.backtest_weeks)
    accuracy = accuracy_summary(bt)

    logger.info("Forecasting %d weeks ahead for every product...", settings.horizon_weeks)
    forecast = forecast_all_products(weekly, horizon_weeks=settings.horizon_weeks)

    logger.info("Solving the production capacity LP for every line/week...")
    production_plan = build_production_plan(
        forecast,
        tables["dim_product"],
        tables["dim_production_line"],
        tables["fact_workforce_availability"],
        operators_per_running_hour=settings.operators_per_running_hour,
        hours_per_shift=settings.hours_per_shift,
        oee=settings.oee,
    )

    logger.info("Running MRP: exploding the plan through the BOM and netting against inventory...")
    requirements = explode_requirements(production_plan, tables["bom"])
    on_hand = latest_on_hand(tables["fact_inventory_transactions"])
    mrp = net_requirements_and_recommend_orders(requirements, tables["dim_raw_material"], on_hand)

    weekly_summary = build_weekly_summary(production_plan, tables["dim_product"], mrp)

    logger.info("Computing the cash conversion cycle (DIO / DSO / DPO) time series...")
    ccc = compute_ccc_timeseries(
        tables["fact_sales_history"],
        tables["fact_purchase_orders"],
        tables["fact_inventory_transactions"],
        tables["dim_region"],
        tables["dim_supplier"],
        tables["dim_raw_material"],
    )
    avg_ccc_days = float(ccc["ccc"].tail(12).mean()) if not ccc.empty else float("nan")

    quality = run_quality_checks(weekly_summary, accuracy, settings.quality)
    if not quality.passed and fail_on_quality:
        raise RuntimeError(f"Quality gate failed: {'; '.join(quality.failures)}")

    write_outputs(
        {
            "forecast": forecast,
            "production_plan": production_plan,
            "material_requirements": mrp,
            "weekly_summary": weekly_summary,
            "backtest_accuracy": accuracy,
            "working_capital": ccc,
        },
        settings.processed_dir,
        settings.parquet_dir,
    )

    duration = time.monotonic() - started
    logger.info("Pipeline completed in %.2fs", duration)

    return PipelineResult(
        weeks_forecasted=settings.horizon_weeks,
        total_forecast_units=int(forecast["forecast_units"].sum()),
        total_allocated_units=int(production_plan["allocated_units"].sum()),
        avg_ccc_days=avg_ccc_days,
        quality=quality,
        duration_seconds=duration,
    )
