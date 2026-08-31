import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest

from flagquantum.testing import (
    MPSCapacityCertificationError,
    require_capacity_source_integrity,
    require_general_mps_capacity,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "benchmarks/development/issue092_general_mps_capacity.json"
A800_ARTIFACT = (
    ROOT
    / "benchmarks/results/local/mps_capacity_12288q_chi768_8xa800_complete_20260805.json"
)
DUAL_NODE_ARTIFACT = (
    ROOT
    / "benchmarks/results/local/mps_capacity_24576q_chi768_16xa800_complete_20260805.json"
)
TOP_CAPACITY_ARTIFACT = (
    ROOT
    / "benchmarks/results/local/mps_capacity_65536q_chi768_16xa800_complete_20260805.json"
)
EXTREME_CAPACITY_ARTIFACT = (
    ROOT
    / "benchmarks/results/local/mps_capacity_98304q_chi768_16xa800_complete_20260805.json"
)
BOUNDARY_CAPACITY_ARTIFACT = (
    ROOT
    / "benchmarks/results/local/mps_capacity_114688q_chi768_16xa800_complete_20260805.json"
)
pytestmark = pytest.mark.benchmark_contract


@pytest.mark.skipif(
    not ARTIFACT.exists(), reason="legacy ISSUE-092 evidence is not present"
)
def test_issue092_legacy_artifact_is_rejected_until_remeasured():
    payload = json.loads(ARTIFACT.read_text())
    with pytest.raises(MPSCapacityCertificationError):
        require_general_mps_capacity(payload)


@pytest.mark.skipif(
    not A800_ARTIFACT.exists(), reason="A800 ISSUE-092 evidence is not present"
)
def test_issue092_a800_capacity_artifact_is_complete_and_sealed():
    payload = json.loads(A800_ARTIFACT.read_text())
    require_general_mps_capacity(payload)
    require_capacity_source_integrity(payload, base_dir=ROOT)
    assert payload["world_size"] == 8
    assert payload["n_sites"] == 12_288
    assert payload["initial_max_bond"] == 768
    assert payload["logical_mps_bytes"] > 80 << 30
    assert payload["discarded_weight"] <= payload["truncation_error_budget"]
    assert len(payload["boundary_evidence"]) == 7
    assert all(record["cleanup_verified"] for record in payload["rank_records"])


@pytest.mark.skipif(
    not DUAL_NODE_ARTIFACT.exists(), reason="16×A800 capacity evidence is not present"
)
def test_issue092_dual_node_capacity_exceeds_two_hundred_gibibytes():
    payload = json.loads(DUAL_NODE_ARTIFACT.read_text())
    require_general_mps_capacity(payload)
    require_capacity_source_integrity(payload, base_dir=ROOT)
    assert payload["world_size"] == 16
    assert payload["n_sites"] == 24_576
    assert payload["initial_max_bond"] == 768
    assert payload["logical_mps_bytes"] > 200 << 30
    assert len(payload["boundary_evidence"]) == 15
    assert payload["discarded_weight"] <= payload["truncation_error_budget"]
    assert all(record["cleanup_verified"] for record in payload["rank_records"])


@pytest.mark.skipif(
    not TOP_CAPACITY_ARTIFACT.exists(), reason="65,536-site evidence is not present"
)
def test_issue092_dual_node_capacity_exceeds_five_hundred_gibibytes():
    payload = json.loads(TOP_CAPACITY_ARTIFACT.read_text())
    require_general_mps_capacity(payload)
    require_capacity_source_integrity(payload, base_dir=ROOT)
    assert payload["world_size"] == 16
    assert payload["n_sites"] == 65_536
    assert payload["initial_max_bond"] == 768
    assert payload["logical_mps_bytes"] > 500 << 30
    assert len(payload["boundary_evidence"]) == 15
    assert payload["discarded_weight"] <= payload["truncation_error_budget"]
    assert all(record["cleanup_verified"] for record in payload["rank_records"])


@pytest.mark.skipif(
    not EXTREME_CAPACITY_ARTIFACT.exists(), reason="98,304-site evidence is absent"
)
def test_issue092_dual_node_capacity_exceeds_eight_hundred_gibibytes():
    payload = json.loads(EXTREME_CAPACITY_ARTIFACT.read_text())
    require_general_mps_capacity(payload)
    require_capacity_source_integrity(payload, base_dir=ROOT)
    assert payload["world_size"] == 16
    assert payload["n_sites"] == 98_304
    assert payload["logical_mps_bytes"] > 800 << 30
    assert len(payload["boundary_evidence"]) == 15
    assert payload["discarded_weight"] <= payload["truncation_error_budget"]
    assert all(record["cleanup_verified"] for record in payload["rank_records"])


@pytest.mark.skipif(
    not BOUNDARY_CAPACITY_ARTIFACT.exists(), reason="114,688-site evidence is absent"
)
def test_issue092_dual_node_capacity_exceeds_one_thousand_gibibytes():
    payload = json.loads(BOUNDARY_CAPACITY_ARTIFACT.read_text())
    require_general_mps_capacity(payload)
    require_capacity_source_integrity(payload, base_dir=ROOT)
    assert payload["world_size"] == 16
    assert payload["n_sites"] == 114_688
    assert payload["logical_mps_bytes"] > 1000 << 30
    assert len(payload["boundary_evidence"]) == 15
    assert payload["discarded_weight"] <= payload["truncation_error_budget"]
    assert all(record["cleanup_verified"] for record in payload["rank_records"])


@pytest.mark.skipif(
    not DUAL_NODE_ARTIFACT.exists(), reason="16×A800 capacity evidence is not present"
)
def test_issue092_compiled_layer_has_measured_multi_owner_power_activity():
    payload = json.loads(DUAL_NODE_ARTIFACT.read_text())
    telemetry = ROOT / next(
        item["path"]
        for item in payload["source_artifacts"]
        if item["kind"] == "gpu_samples"
    )
    samples: dict[str, list[float]] = defaultdict(list)
    per_gpu_peak: dict[int, float] = defaultdict(float)
    with telemetry.open(newline="") as stream:
        for row in csv.DictReader(stream):
            power = float(row[" power.draw [W]"].strip().removesuffix(" W"))
            index = int(row[" index"])
            samples[row["timestamp"]].append(power)
            per_gpu_peak[index] = max(per_gpu_peak[index], power)
    # Two simultaneous high-power owners proves multi-owner execution.  Do not
    # reward extra power draw after sparse-adjoint elimination removes useless
    # per-site autograd work; end-to-end latency is the efficiency gate.
    assert max(sum(power > 180 for power in values) for values in samples.values()) >= 2
    assert max(record["elapsed_seconds"] for record in payload["rank_records"]) < 75.0
    assert set(per_gpu_peak) == set(range(8))
    assert min(per_gpu_peak.values()) > 200
