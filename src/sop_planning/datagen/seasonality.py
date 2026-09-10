"""Demand seasonality curves.

The whole point of using an industrial cake & confectionery food manufacturer producing Persian
sweets (kolucheh) instead of a generic "widget factory" is that it comes
with genuine, well-known seasonal demand spikes — Nowruz (Persian New Year)
and Yalda Night chief among them — which makes the forecasting problem
realistic instead of a flat, noise-only series.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# Nowruz always falls close to March 20-21; Yalda is the winter solstice,
# ~December 21. Ramadan is lunar and drifts ~11 days/year — approximate
# windows for the years this dataset covers are hardcoded below rather than
# computed from the Hijri calendar, which is precise enough for a synthetic
# demand curve.
NOWRUZ_DATES = {2024: date(2024, 3, 20), 2025: date(2025, 3, 20), 2026: date(2026, 3, 21)}
YALDA_DATES = {2024: date(2024, 12, 21), 2025: date(2025, 12, 21), 2026: date(2026, 12, 21)}
RAMADAN_WINDOWS = {
    2024: (date(2024, 3, 11), date(2024, 4, 9)),
    2025: (date(2025, 3, 1), date(2025, 3, 30)),
}


def _gaussian_bump(days_from_peak: np.ndarray, width_days: float, magnitude: float) -> np.ndarray:
    return 1.0 + magnitude * np.exp(-0.5 * (days_from_peak / width_days) ** 2)


def seasonal_multiplier(dates: pd.DatetimeIndex, seasonal_peak: str | None) -> np.ndarray:
    """Return a per-date multiplier (1.0 = baseline demand) for a product."""
    mult = np.ones(len(dates))
    d = dates.date

    if seasonal_peak == "Nowruz":
        for year, peak in NOWRUZ_DATES.items():
            days_from = np.array([(dd - peak).days for dd in d])
            # Demand builds up in the ~18 days before Nowruz and collapses after.
            pre = (days_from >= -18) & (days_from <= 0)
            mult[pre] *= 1.0 + 7.0 * np.exp(-0.5 * (days_from[pre] / 8.0) ** 2)
    elif seasonal_peak == "Yalda":
        for year, peak in YALDA_DATES.items():
            days_from = np.array([(dd - peak).days for dd in d])
            near = np.abs(days_from) <= 10
            mult[near] *= _gaussian_bump(days_from[near], width_days=4.0, magnitude=4.5)
    elif seasonal_peak == "Ramadan":
        for year, (start, end) in RAMADAN_WINDOWS.items():
            in_window = np.array([(start <= dd <= end) for dd in d])
            mult[in_window] *= 3.0
    elif seasonal_peak == "Summer":
        month = dates.month.values
        mult *= np.where(np.isin(month, [6, 7, 8]), 1.8, 1.0)
    elif seasonal_peak == "AutumnStart":
        # Iranian school year starts ~Sept 23 (Mehr 1).
        month, day = dates.month.values, dates.day.values
        in_window = ((month == 9) & (day >= 15)) | ((month == 10) & (day <= 10))
        mult *= np.where(in_window, 3.2, 1.0)

    return mult


def weekday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Iran's weekend is Friday (with Thursday often a shorter day) —
    cake/confectionery demand skews toward the end of the week."""
    weekday = dates.dayofweek.values  # Monday=0 ... Sunday=6
    mult = np.ones(len(dates))
    mult[weekday == 3] = 1.15  # Thursday
    mult[weekday == 4] = 1.35  # Friday
    mult[weekday == 5] = 0.95  # Saturday (first day of Iranian work week)
    return mult
