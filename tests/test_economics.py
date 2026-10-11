"""Waste, plant economics and the value ledger: identities that must hold for any seed."""

import pytest

from sop_planning.analytics import bullwhip, economics

RAW = "data/raw"
DB = "data/processed/planning_warehouse.db"


@pytest.fixture(scope="module")
def waste():
    return economics.waste_detail(RAW)


@pytest.fixture(scope="module")
def econ():
    return economics.plant_economics(RAW, DB)


def test_scrap_savings_never_exceed_scrap_cost(waste):
    assert 0 <= waste["kpi"]["save_year"] <= waste["kpi"]["cost_year"]
    assert all(p["save_year"] <= p["cost_year"] for p in waste["top_products"])


def test_scrap_rates_are_realistic(waste):
    assert 0.005 < waste["kpi"]["rate"] < 0.08


def test_pnl_adds_up(econ):
    k = econ["kpi"]
    assert k["contribution"] == pytest.approx(k["revenue"] - k["materials"] - k["payroll"])
    assert 0.2 < k["materials_pct"] < 0.6  # food manufacturing: materials are 30-55% of revenue
    assert sum(line["revenue"] for line in econ["lines"]) == pytest.approx(k["revenue"])


def test_ledger_values_are_positive(waste, econ):
    ledger = economics.value_ledger(
        bullwhip.summarize(RAW), waste, econ, {"bottleneck_margin": 1000.0}
    )
    assert {x["kind"] for x in ledger} == {"cash", "annual"}
    assert all(x["value"] >= 0 for x in ledger)
