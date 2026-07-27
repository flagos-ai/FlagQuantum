"""Fail-closed contracts for the three-way system-identification capacity case."""

import importlib.util
from pathlib import Path

import pytest

MODULE = Path("benchmarks/mps_system_id_capacity_case.py")
SPEC = importlib.util.spec_from_file_location("mps_system_id_capacity_case", MODULE)
capacity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(capacity)

pytestmark = pytest.mark.benchmark_contract


def _sharded_result():
    return {
        "distribution_semantics": "sharded_across_ranks",
        "teacher_full_mps_materialization": False,
        "optimizer_steps": 1,
        "best_checkpoints": ["rank-0.pt", "rank-1.pt"],
        "gpu_evidence": {
            "accelerator": True,
            "peak_memory_bytes_by_rank": [10, 11],
        },
        "communication": {
            "backend": "nccl",
            "measured_logical_boundary_bytes": 128,
        },
    }


def test_capacity_gate_requires_both_full_mps_capacity_failures():
    single = {"status": "cuda_oom", "workload_sha256": "same"}
    replicated = {"status": "cuda_oom", "workload_sha256": "same"}
    sharded = {
        "status": "passed",
        "workload_sha256": "same",
        "result": _sharded_result(),
    }
    passed, blockers = capacity.capacity_gate(single, replicated, sharded)
    assert passed is True
    assert blockers == ()


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        (
            ("replicated", "status", "passed"),
            "replicated_capacity_failure_not_measured",
        ),
        (("sharded", "status", "cuda_oom"), "site_sharded_training_not_completed"),
        (("result", "optimizer_steps", 0), "adam_update_not_completed"),
        (
            ("result", "teacher_full_mps_materialization", True),
            "teacher_full_mps_materialization_not_rejected",
        ),
    ],
)
def test_capacity_gate_fails_closed(mutation, blocker):
    records = {
        "single": {"status": "cuda_oom", "workload_sha256": "same"},
        "replicated": {"status": "cuda_oom", "workload_sha256": "same"},
        "sharded": {
            "status": "passed",
            "workload_sha256": "same",
            "result": _sharded_result(),
        },
    }
    target, key, value = mutation
    if target == "result":
        records["sharded"]["result"][key] = value
    else:
        records[target][key] = value
    passed, blockers = capacity.capacity_gate(
        records["single"], records["replicated"], records["sharded"]
    )
    assert passed is False
    assert blocker in blockers


def test_failure_classifier_distinguishes_capacity_from_other_failures():
    assert capacity.classify_failure(1, "CUDA out of memory") == "cuda_oom"
    assert capacity.classify_failure(124, "timed out") == "timeout"
    assert capacity.classify_failure(1, "shape mismatch") == "runtime_error"
