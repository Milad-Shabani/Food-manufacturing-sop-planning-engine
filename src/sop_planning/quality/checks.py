"""Quality gate for the planning run: catches a plan that's obviously
broken (can't meet a reasonable fraction of demand, or a forecast that's
backtesting far worse than it should) before it's published.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..config import QualityThresholds

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    min_weekly_fill_rate: float
    overall_backtest_wape: float
    passed: bool
    failures: list[str]


def run_quality_checks(
    weekly_summary: pd.DataFrame, backtest_accuracy: pd.DataFrame, thresholds: QualityThresholds
) -> QualityReport:
    min_fill_rate = float(weekly_summary["fill_rate"].min()) if not weekly_summary.empty else 1.0

    overall_row = backtest_accuracy[backtest_accuracy["product_id"] == "__OVERALL__"]
    overall_wape = float(overall_row["wape"].iloc[0]) if not overall_row.empty else 0.0

    failures = []
    if min_fill_rate < thresholds.min_fill_rate:
        failures.append(
            f"minimum weekly fill rate {min_fill_rate:.1%} below threshold "
            f"{thresholds.min_fill_rate:.1%}"
        )
    if overall_wape > thresholds.max_backtest_wape:
        failures.append(
            f"backtest WAPE {overall_wape:.1%} exceeds threshold {thresholds.max_backtest_wape:.1%}"
        )

    if failures:
        logger.error("Quality checks FAILED: %s", "; ".join(failures))
    else:
        logger.info(
            "Quality checks passed (min_fill_rate=%.1f%%, backtest_wape=%.1f%%)",
            min_fill_rate * 100,
            overall_wape * 100,
        )

    return QualityReport(min_fill_rate, overall_wape, not failures, failures)
