"""Immutable historical receipts and complete paired AO-02 resource evidence."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.synthesis.normalization import FROZEN_SHA256, differences
from benchmarks.synthesis.results import read_result

pytestmark = pytest.mark.benchmark_contract
BASELINES = Path(__file__).resolve().parents[2] / "benchmarks/synthesis/baselines"


def test_original_and_intermediate_baselines_and_oracles_stay_immutable() -> None:
    assert (
        hashlib.sha256((BASELINES / "ao00.json").read_bytes()).hexdigest()
        == FROZEN_SHA256
    )
    assert (
        hashlib.sha256((BASELINES / "ao01.json").read_bytes()).hexdigest()
        == "e54047e626c60c132d902d7007ff59f55ca8b2df99e8530988dad9805fca426d"
    )
    current = read_result(BASELINES / "ao02.json")
    for name in ("ao00.json", "ao01.json"):
        original = read_result(BASELINES / name)
        assert current["configuration"] == original["configuration"]
        assert all(
            current["source_hashes"][key] == value
            for key, value in original["source_hashes"].items()
        )


def test_fixed_matrix_comparison_keeps_timeouts_incomplete_and_reproduces_counts() -> (
    None
):
    rows = differences(BASELINES / "ao00.json", BASELINES / "ao02.json")
    receipt = json.loads((BASELINES / "ao02-comparison.json").read_text())
    assert receipt["rows"] == [asdict(row) for row in rows]
    assert receipt["summary"] == dict(Counter(row.classification for row in rows))
    assert not any(row.classification in {"regressed", "tradeoff"} for row in rows)
    assert len(rows) == 90
    assert sum(row.classification == "incomplete" for row in rows) == 1


def test_pairs_show_multiple_actual_gains_for_every_rule_and_full_target_checks() -> (
    None
):
    receipt = json.loads((BASELINES / "ao02-paired.json").read_text())
    assert receipt["schema"] == "flagquantum.synthesis.arithmetic_implementations.v1"
    assert (
        receipt["source_revision"]
        == read_result(BASELINES / "ao02.json")["source_revision"]
    )
    rows = receipt["rows"]
    assert len(rows) == 16
    groups: dict[str, set[str]] = {}
    for row in rows:
        old, new = row["conservative"], row["selected"]
        keys = ("allocated_qubits", "workspace_qubits", "operations", "depth")
        assert all(new[key] <= old[key] for key in keys)
        assert any(new[key] < old[key] for key in keys)
        assert new["operations"] == sum(new["operation_counts"].values())
        assert new["planned_complex128_state_bytes"] == 16 << new["allocated_qubits"]
        assert row["exhaustive_input_target_checks"] > 0
        for rule in row["rules"]:
            groups.setdefault(rule, set()).add(row["case"])
    assert sum(row["exhaustive_input_target_checks"] for row in rows) == 16512
    assert all(
        len(groups[rule]) >= 2
        for rule in ("square_symmetry", "constant_shift_add", "terminal_xor_add")
    )
