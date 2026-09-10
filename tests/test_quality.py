import pandas as pd

from sop_planning.config import QualityThresholds
from sop_planning.quality.checks import run_quality_checks


def test_quality_passes_within_thresholds():
    weekly_summary = pd.DataFrame({"week_start": ["2026-01-04"], "fill_rate": [0.95]})
    accuracy = pd.DataFrame(
        [{"product_id": "__OVERALL__", "mape": None, "wape": 0.2, "n_weeks": 8}]
    )
    thresholds = QualityThresholds(min_fill_rate=0.80, max_backtest_wape=0.45)

    report = run_quality_checks(weekly_summary, accuracy, thresholds)

    assert report.passed
    assert not report.failures


def test_quality_fails_when_fill_rate_too_low():
    weekly_summary = pd.DataFrame({"week_start": ["2026-01-04"], "fill_rate": [0.5]})
    accuracy = pd.DataFrame(
        [{"product_id": "__OVERALL__", "mape": None, "wape": 0.2, "n_weeks": 8}]
    )
    thresholds = QualityThresholds(min_fill_rate=0.80, max_backtest_wape=0.45)

    report = run_quality_checks(weekly_summary, accuracy, thresholds)

    assert not report.passed
    assert any("fill rate" in f for f in report.failures)


def test_quality_fails_when_backtest_wape_too_high():
    weekly_summary = pd.DataFrame({"week_start": ["2026-01-04"], "fill_rate": [1.0]})
    accuracy = pd.DataFrame(
        [{"product_id": "__OVERALL__", "mape": None, "wape": 0.9, "n_weeks": 8}]
    )
    thresholds = QualityThresholds(min_fill_rate=0.80, max_backtest_wape=0.45)

    report = run_quality_checks(weekly_summary, accuracy, thresholds)

    assert not report.passed
    assert any("WAPE" in f for f in report.failures)
