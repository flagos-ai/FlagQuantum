import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.internal.evidence.statevector_release_gate import (
    RESULTS,
    baseline_results,
    evaluate_issue044_release,
    load_manifest,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]


def _artifact(
    world_size: int,
    acceptance_case: str,
    *,
    node_count: int = 1,
) -> dict[str, object]:
    manifest = load_manifest()
    workload_sha256 = manifest["capacity_workload"]["workload_sha256"]
    baseline = world_size == 1
    evidence = {
        "release_gate_allowed": not baseline,
        "acceptance_case": acceptance_case,
        "world_size": world_size,
        "node_count": node_count,
        "workload_sha256": workload_sha256,
        "distribution_semantics": "sharded_across_ranks",
        "speedup": 1.2,
        "speedup_confidence_interval": [1.1, 1.3],
        "scaling_efficiency": 0.6,
        "single_device_oom_observed": acceptance_case == "single_gpu_capacity_failure",
        "capacity_baseline_device": "cuda:0",
        "capacity_failure_reason": "CUDA out of memory",
        "measured_peak_memory_bytes": 68728000512,
        "training_step_count": 2,
        "full_state_materialized": False,
    }
    if not baseline:
        evidence.update(
            {
                "rank_outputs": [0.0] * world_size,
                "rank_gradients": [0.0] * world_size,
                "rank_timings": [1.0] * world_size,
                "rank_peak_memory_bytes": [1024] * world_size,
                "communication_fraction": 0.1,
                "gpu_activity": 0.8,
            }
        )
    return {
        "schema": "flagquantum_runtime_evidence_v1",
        "artifact_class": "measured_production_run",
        "evidence_scope": ("one_gpu_local" if baseline else "scheduled_4_8_gpu_scale"),
        "provenance": {
            "commit": "a" * 40,
            "workload_sha256": workload_sha256,
            "command": ["torchrun"],
            "devices": [f"GPU-{rank}" for rank in range(world_size)],
            "topology": "test topology",
            "rank_mapping": [f"rank={rank}" for rank in range(world_size)],
            "collective_backend": "nccl",
            "warmup": 5,
            "iterations": 30,
            "seeds": [440044],
            "raw_log_sha256": "b" * 64,
            "fallback_events": [],
        },
        "evidence": evidence,
        "integrity": {},
    }


def _passing_artifacts() -> list[dict[str, object]]:
    return [
        _artifact(1, "single_gpu_capacity_failure"),
        _artifact(2, "matched_speed"),
        _artifact(4, "multi_gpu_capacity_completion"),
        _artifact(8, "multinode_correctness", node_count=2),
    ]


def test_manifest_freezes_workloads_thresholds_and_topologies():
    manifest = load_manifest()
    assert manifest["topologies"]["single_node_world_sizes"] == [2, 4, 8]
    assert manifest["topologies"]["multi_node_required"] is True
    assert manifest["speed_workload"]["minimum_speedup"] > 1
    assert manifest["speed_workload"]["measured_steps"] >= 30
    assert manifest["capacity_workload"]["single_gpu_measured_oom_required"] is True
    assert manifest["runtime"]["backend"] == "pytorch_native"
    capacity = manifest["capacity_workload"]
    workload_path = Path(capacity["manifest_path"])
    assert capacity["n_wires"] == 31
    assert (
        hashlib.sha256(workload_path.read_bytes()).hexdigest()
        == capacity["workload_sha256"]
    )


def test_single_device_baseline_is_provenance_evidence_not_a_release_world():
    manifest = load_manifest()
    baseline = manifest["capacity_workload"]["single_gpu_baseline"]

    assert baseline["release_gate_payload"] is False
    assert baseline["evidence_scope"] == "one_gpu_local"
    assert 1 not in manifest["topologies"]["single_node_world_sizes"]


def test_baseline_directory_stays_outside_the_promoted_release_directory():
    """The baseline cannot be promoted, so it cannot live in the promoted set."""

    path = baseline_results(load_manifest())

    assert path != RESULTS
    assert RESULTS not in path.parents
    assert path.is_relative_to("benchmarks/results")


def test_a_baseline_directory_under_the_promoted_set_is_rejected():
    manifest = load_manifest()
    manifest["capacity_workload"]["single_gpu_baseline"]["results_path"] = str(
        RESULTS / "statevector_single_gpu_capacity"
    )

    with pytest.raises(ValueError, match="cannot live under"):
        baseline_results(manifest)


def test_single_device_world_cannot_pose_as_a_release_payload():
    artifacts = _passing_artifacts()
    artifacts[0]["evidence"]["release_gate_allowed"] = True

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "single_device_world_cannot_be_a_release_payload" in blockers


def test_baseline_outside_one_gpu_local_scope_cannot_satisfy_the_capacity_case():
    artifacts = deepcopy(_passing_artifacts())
    artifacts[0]["evidence_scope"] = "scheduled_4_8_gpu_scale"

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "missing_single_gpu_measured_oom_artifact" in blockers


def test_baseline_missing_a_declared_field_is_rejected():
    artifacts = deepcopy(_passing_artifacts())
    del artifacts[0]["evidence"]["capacity_baseline_device"]

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "single_gpu_baseline_missing_required_fields" in blockers


def test_empty_release_directory_fails_closed():
    passed, blockers = evaluate_issue044_release([], load_manifest())
    assert passed is False
    assert "missing_release_world_sizes" in blockers


def test_sealed_envelope_field_layout_can_pass_gate():
    passed, blockers = evaluate_issue044_release(_passing_artifacts(), load_manifest())

    assert passed is True
    assert blockers == ()


def test_named_speed_case_cannot_bypass_frozen_statistical_thresholds():
    artifacts = deepcopy(_passing_artifacts())
    artifacts[1]["evidence"]["speedup_confidence_interval"] = [0.99, 1.4]

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "missing_statistically_significant_speedup_artifact" in blockers


def test_flat_unsigned_production_payload_is_rejected():
    artifact = {
        "artifact_class": "measured_production_run",
        "release_gate_allowed": True,
        "world_size": 8,
    }

    passed, blockers = evaluate_issue044_release([artifact], load_manifest())

    assert passed is False
    assert "invalid_or_unsigned_production_artifact" in blockers


def test_malformed_numeric_measurements_fail_closed_without_crashing():
    artifacts = _passing_artifacts()
    artifacts[1]["evidence"]["speedup"] = None
    artifacts[2]["evidence"]["training_step_count"] = "unknown"

    passed, blockers = evaluate_issue044_release(artifacts, load_manifest())

    assert passed is False
    assert "missing_statistically_significant_speedup_artifact" in blockers
    assert "missing_multi_gpu_capacity_completion_artifact" in blockers
