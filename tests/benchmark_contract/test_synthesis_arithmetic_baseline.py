"""AO-00 artifacts keep unexecuted domains separate from quantum evidence."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.synthesis.arithmetic_baseline import bounded_measure
from benchmarks.synthesis.expressions import EXPRESSIONS, Case
from benchmarks.synthesis.results import compare, read_result

pytestmark = pytest.mark.benchmark_contract
BASELINE = (
    Path(__file__).resolve().parents[2] / "benchmarks/synthesis/baselines/ao00.json"
)


def test_checked_baseline_and_repeat_ignore_observational_timing(
    tmp_path: Path,
) -> None:
    payload = read_result(BASELINE)
    repeated = deepcopy(payload)
    repeated["environment"] = {"different_host": True}
    repeated["source_revision"] = "f" * 40
    rows = repeated["rows"]
    assert isinstance(rows, list)
    for row in rows:
        row["synthesis_and_planning_seconds"] = 999.0
    target = tmp_path / "repeat.json"
    target.write_text(json.dumps(repeated))
    assert compare(BASELINE, target) == ()


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "domain",
        "width",
        "missing_case",
        "memory",
        "histogram",
        "extra_resource",
        "static_quantum_claim",
        "summary",
        "incomplete_cost_claim",
    ],
)
def test_result_reader_rejects_corrupted_or_overstated_evidence(
    tmp_path: Path, mutation: str
) -> None:
    payload = json.loads(BASELINE.read_text())
    row = payload["rows"][0]
    if mutation == "schema":
        payload["schema"] = "unknown"
    elif mutation == "domain":
        row["widths"] = [2, 2]
    elif mutation == "width":
        row["output_width"] = 1
    elif mutation == "missing_case":
        payload["rows"].pop()
    elif mutation == "memory":
        row["resources"]["planned_complex128_state_bytes"] += 1
    elif mutation == "histogram":
        row["resources"]["operations"] += 1
    elif mutation == "extra_resource":
        row["resources"]["new_metric"] = 0
    elif mutation == "static_quantum_claim":
        row["verification"]["coherent_states"] = 0
    elif mutation == "summary":
        payload["summary"]["compiled"] += 1
    elif mutation == "incomplete_cost_claim":
        row["status"] = "timeout"
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_result(corrupted)


def test_different_configuration_is_not_comparable(tmp_path: Path) -> None:
    payload = json.loads(BASELINE.read_text())
    payload["configuration"]["objective"] = "depth"
    changed = tmp_path / "depth.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="configuration"):
        compare(BASELINE, changed)


def test_process_timeout_never_claims_infeasibility_costs_or_verification() -> None:
    row = bounded_measure(Case(EXPRESSIONS[4], (16, 16)), timeout_seconds=0.001)
    assert row["status"] == "timeout"
    assert row["diagnostic_codes"] == ("benchmark.wall_timeout",)
    assert row["resources"] is row["verification"] is None
    assert row["synthesis_and_planning_seconds"] is None
    assert row["worker_wall_seconds"] > 0
