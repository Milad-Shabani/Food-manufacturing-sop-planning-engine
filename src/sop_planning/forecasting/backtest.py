"""Rolling backtest: measure forecast accuracy on held-out history.

For each product, the last `test_weeks` of known weekly demand are hidden,
a forecast is produced from everything before that point using the same
`forecast_all_products` logic, and the forecast is compared to what
actually happened. This is what lets the README claim a real accuracy
number instead of an unverified one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecaster import forecast_all_products


def backtest(weekly: pd.DataFrame, test_weeks: int = 8) -> pd.DataFrame:
    rows = []
    for product_id, group in weekly.groupby("product_id"):
        group = group.sort_values("week_start")
        if len(group) <= test_weeks + 8:
            continue  # not enough history to backtest meaningfully

        train = group.iloc[:-test_weeks]
        actual = group.iloc[-test_weeks:]

        forecast = forecast_all_products(
            train.assign(product_id=product_id), horizon_weeks=test_weeks
        )
        merged = forecast.merge(
            actual[["week_start", "demand_units"]], on="week_start", how="inner"
        )
        merged["product_id"] = product_id
        rows.append(merged)

    if not rows:
        return pd.DataFrame(columns=["product_id", "week_start", "forecast_units", "demand_units"])
    return pd.concat(rows, ignore_index=True)


def accuracy_summary(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """MAPE and WAPE per product, plus a company-wide WAPE.

    WAPE (weighted absolute percentage error) is reported alongside MAPE
    because MAPE blows up / becomes meaningless for low-volume weeks (a
    product selling 3 units one week can produce a triple-digit % error
    that says nothing useful); WAPE is far more stable for that case and is
    the metric a real S&OP process would actually track.
    """
    if backtest_df.empty:
        return pd.DataFrame(columns=["product_id", "mape", "wape", "n_weeks"])

    def _wape(g: pd.DataFrame) -> float:
        denom = g["demand_units"].sum()
        return (
            float(np.abs(g["forecast_units"] - g["demand_units"]).sum() / denom)
            if denom
            else np.nan
        )

    def _mape(g: pd.DataFrame) -> float:
        nonzero = g[g["demand_units"] > 0]
        if nonzero.empty:
            return np.nan
        return float(
            (
                np.abs(nonzero["forecast_units"] - nonzero["demand_units"])
                / nonzero["demand_units"]
            ).mean()
        )

    per_product = (
        backtest_df.groupby("product_id")
        .apply(lambda g: pd.Series({"mape": _mape(g), "wape": _wape(g), "n_weeks": len(g)}))
        .reset_index()
    )

    overall_wape = _wape(backtest_df)
    overall = pd.DataFrame(
        [
            {
                "product_id": "__OVERALL__",
                "mape": np.nan,
                "wape": overall_wape,
                "n_weeks": len(backtest_df),
            }
        ]
    )
    return pd.concat([per_product, overall], ignore_index=True)
