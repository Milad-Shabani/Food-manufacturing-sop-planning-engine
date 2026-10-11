"""Line and warehouse detail: identities that must hold regardless of the seed."""

import pytest

from sop_planning.analytics import operations

RAW = "data/raw"
DB = "data/processed/planning_warehouse.db"


@pytest.fixture(scope="module")
def lines():
    return operations.line_detail(RAW, DB)


@pytest.fixture(scope="module")
def wh():
    return operations.warehouse_detail(RAW)


def test_oee_is_product_of_its_parts(lines):
    for ln in lines["lines"]:
        assert ln["oee"] == pytest.approx(ln["availability"] * ln["performance"] * ln["quality"])
        assert 0 < ln["oee"] <= 1.0


def test_utilization_never_exceeds_the_lp_limit(lines):
    for row in lines["heat"]:
        assert all(0 <= u <= 1.0001 for u in row["util"])


def test_warehouse_classes_and_shares(wh):
    assert sum(m["share"] for m in wh["materials"]) == pytest.approx(1.0)
    assert {m["abc"] for m in wh["materials"]} <= {"A", "B", "C"}
    assert wh["pareto"][-1]["cum"] == pytest.approx(1.0)
    assert sum(s["spend_share"] for s in wh["suppliers"]) == pytest.approx(1.0)
