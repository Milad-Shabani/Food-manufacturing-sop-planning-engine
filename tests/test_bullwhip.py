"""Bullwhip layer: the chain only ever adds stock; the managed policy ties up less cash."""

import pandas as pd
import pytest

from sop_planning.analytics import bullwhip

RAW = "data/raw"


@pytest.fixture(scope="module")
def chain():
    return bullwhip.simulate_chain(RAW, seed=42)


def test_layers_are_monotonic(chain):
    assert (chain.supervisor_qty >= chain.need_qty).all()
    assert (chain.manager_qty >= chain.supervisor_qty).all()
    assert (chain.procurement_qty >= chain.manager_qty).all()
    assert (chain.purchase_qty >= chain.procurement_qty).all()


def test_managed_policy_ties_up_less_cash(chain):
    assert chain.managed_excess_value_usd.sum() < chain.excess_value_usd.sum()
    assert (chain.managed_qty >= chain.need_qty).all()  # never under-buys the real need


def test_deterministic_for_a_seed():
    a = bullwhip.simulate_chain(RAW, seed=7).purchase_qty
    b = bullwhip.simulate_chain(RAW, seed=7).purchase_qty
    pd.testing.assert_series_equal(a, b)


def test_summary_shape():
    s = bullwhip.summarize(RAW, seed=42)
    assert s["stack"]["need"] == pytest.approx(100.0)
    assert s["stack"]["purchase"] > s["stack"]["procurement"] > s["stack"]["manager"] > 100
    assert s["kpi"]["freed_cash"] > 0
