"""Configuration loading: config/config.yaml for pipeline settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QualityThresholds:
    min_fill_rate: float = 0.80
    max_backtest_wape: float = 0.45


@dataclass
class PlanningSettings:
    raw_dir: str
    processed_dir: str
    parquet_dir: str
    horizon_weeks: int
    backtest_weeks: int
    operators_per_running_hour: dict[str, float]
    hours_per_shift: float
    oee: float
    quality: QualityThresholds = field(default_factory=QualityThresholds)


def load_settings(config_path: str | Path = "config/config.yaml") -> PlanningSettings:
    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return PlanningSettings(
        raw_dir=raw["paths"]["raw_dir"],
        processed_dir=raw["paths"]["processed_dir"],
        parquet_dir=raw["paths"]["parquet_dir"],
        horizon_weeks=raw["forecast"]["horizon_weeks"],
        backtest_weeks=raw["forecast"]["backtest_weeks"],
        operators_per_running_hour=raw["planning"]["operators_per_running_hour"],
        hours_per_shift=raw["planning"]["hours_per_shift"],
        oee=raw["planning"]["oee"],
        quality=QualityThresholds(
            min_fill_rate=raw["quality"]["min_fill_rate"],
            max_backtest_wape=raw["quality"]["max_backtest_wape"],
        ),
    )
