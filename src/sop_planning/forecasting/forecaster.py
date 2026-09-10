"""Per-SKU weekly demand forecasting.

Daily sales are aggregated to weekly (cake/confectionery demand is noisy
day-to-day but the weekly signal is what production and procurement
actually plan against) and forecast with Holt-Winters exponential smoothing
— triple exponential smoothing with an additive trend and a 52-week
seasonal period, which is enough to capture the Nowruz/Yalda cycle without
needing a heavier model. A naive seasonal-average fallback covers products
with too little history for Holt-Winters to fit.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

MIN_WEEKS_FOR_HOLT_WINTERS = 104  # need ~2 full yearly cycles to fit a 52-week season


def weekly_demand(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales to ISO-week totals per product."""
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week_start"] = df["date"].dt.to_period("W-SAT").apply(lambda p: p.start_time)
    weekly = df.groupby(["product_id", "week_start"], as_index=False)["units_sold"].sum()
    return weekly.rename(columns={"units_sold": "demand_units"})


def _forecast_one_series(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    series = series.astype(float)
    if len(series) >= MIN_WEEKS_FOR_HOLT_WINTERS and series.min() >= 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=52,
                    initialization_method="estimated",
                ).fit(optimized=True)
            forecast = model.forecast(horizon_weeks)
            return np.clip(forecast.values, 0, None)
        except Exception as exc:  # noqa: BLE001 - fall back rather than crash the whole run
            logger.warning("Holt-Winters failed (%s); falling back to seasonal-naive.", exc)

    return _seasonal_naive_forecast(series, horizon_weeks)


def _seasonal_naive_forecast(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    """Repeat the same week-of-year value from last year, or the recent
    average if there isn't a full year of history yet."""
    if len(series) >= 52:
        last_cycle = series.values[-52:]
        reps = int(np.ceil(horizon_weeks / 52))
        return np.tile(last_cycle, reps)[:horizon_weeks]
    return np.full(horizon_weeks, series.tail(8).mean() if len(series) else 0.0)


def forecast_all_products(weekly: pd.DataFrame, horizon_weeks: int = 12) -> pd.DataFrame:
    """Forecast the next `horizon_weeks` of weekly demand for every product."""
    results = []
    for product_id, group in weekly.groupby("product_id"):
        group = group.sort_values("week_start")
        series = group.set_index("week_start")["demand_units"]
        last_week = series.index.max()

        values = _forecast_one_series(series, horizon_weeks)
        future_weeks = [last_week + pd.Timedelta(weeks=i) for i in range(1, horizon_weeks + 1)]

        for wk, val in zip(future_weeks, values):
            results.append(
                {"product_id": product_id, "week_start": wk, "forecast_units": max(round(val), 0)}
            )

    return pd.DataFrame(results)
