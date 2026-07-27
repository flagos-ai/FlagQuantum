from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("benchmarks/statevector_weak_scaling_report.py")
sys.path.insert(0, str(SCRIPT.parent.resolve()))
SPEC = importlib.util.spec_from_file_location("statevector_weak_scaling_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _point(world: int) -> dict:
    wires = 8 + (world.bit_length() - 1)
    return {
        "schema_version": "flagquantum.statevector.strong_scaling.v1",
        "artifact_class": "measured_development_run",
        "backend": "nccl",
        "world_size": world,
        "node_count": 2 if world == 4 else 1,
        "workload": {
            "n_wires": wires,
            "depth": 3,
            "gate_count": 24,
            "dtype": "complex64",
        },
        "correctness": {"passed": True},
        "timing": {
            "samples_seconds": [1.0 + world / 100] * 3,
            "coefficient_of_variation": 0.0,
        },
        "rank_timings": [
            {"rank": rank, "local_state_bytes": 2048} for rank in range(world)
        ],
        "rank_peak_memory_bytes": [4096] * world,
        "communication_tiers": {
            "route_classification": (
                "none"
                if world == 1
                else "inter_node_collective"
                if world == 4
                else "intra_node_collective"
            )
        },
    }


def test_build_report_accepts_constant_local_state_matrix() -> None:
    report = MODULE.build_report([_point(1), _point(2), _point(4)])
    assert report["definition"] == "constant_rank_local_amplitudes"
    assert report["source_world_sizes"] == [1, 2, 4]
    assert [point["n_wires"] for point in report["points"]] == [8, 9, 10]
    assert report["local_state_bytes"] == 2048
    assert report["scalability_claim_allowed"] is False


def test_build_report_rejects_nonconstant_local_state() -> None:
    candidate = _point(2)
    candidate["rank_timings"][0]["local_state_bytes"] = 1024
    try:
        MODULE.build_report([_point(1), candidate])
    except ValueError as error:
        assert "rank-local state sizes differ" in str(error)
    else:
        raise AssertionError("nonconstant rank-local state must fail closed")


def test_build_report_rejects_wrong_qubit_progression() -> None:
    candidate = _point(2)
    candidate["workload"]["n_wires"] = 10
    try:
        MODULE.build_report([_point(1), candidate])
    except ValueError as error:
        assert "qubit count" in str(error)
    else:
        raise AssertionError("invalid weak-scaling workload must fail closed")
