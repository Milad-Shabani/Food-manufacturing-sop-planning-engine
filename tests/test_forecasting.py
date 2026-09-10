import numpy as np
import pandas as pd

from sop_planning.forecasting.backtest import accuracy_summary, backtest
from sop_planning.forecasting.forecaster import forecast_all_products, weekly_demand


def _synthetic_weekly(n_weeks: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2023-01-01", periods=n_weeks, freq="W-SUN")
    seasonal = 50 + 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = rng.normal(0, 3, n_weeks)
    demand = np.clip(seasonal + noise, 0, None)
    return pd.DataFrame({"product_id": "P_TEST", "week_start": weeks, "demand_units": demand})


def test_weekly_demand_aggregates_daily_sales():
    sales = pd.DataFrame(
        [
            {"date": "2024-01-01", "product_id": "P1", "units_sold": 10},
            {"date": "2024-01-02", "product_id": "P1", "units_sold": 15},
            {"date": "2024-01-08", "product_id": "P1", "units_sold": 20},
        ]
    )
    weekly = weekly_demand(sales)
    assert weekly["demand_units"].sum() == 45
    assert len(weekly) == 2  # two distinct weeks


def test_forecast_all_products_returns_requested_horizon():
    weekly = _synthetic_weekly()
    forecast = forecast_all_products(weekly, horizon_weeks=12)

    assert len(forecast) == 12
    assert forecast["week_start"].is_monotonic_increasing
    assert (forecast["forecast_units"] >= 0).all()
    # forecast should continue directly after the last observed week
    assert forecast["week_start"].min() == weekly["week_start"].max() + pd.Timedelta(weeks=1)


def test_backtest_produces_matched_forecast_and_actuals():
    weekly = _synthetic_weekly()
    bt = backtest(weekly, test_weeks=8)

    assert len(bt) == 8
    assert set(bt.columns) == {"product_id", "week_start", "forecast_units", "demand_units"}


def test_accuracy_summary_includes_overall_row_and_reasonable_wape():
    weekly = _synthetic_weekly()
    bt = backtest(weekly, test_weeks=8)
    acc = accuracy_summary(bt)

    overall = acc[acc["product_id"] == "__OVERALL__"].iloc[0]
    assert 0 <= overall["wape"] < 1.0  # a smooth synthetic series should forecast reasonably well
