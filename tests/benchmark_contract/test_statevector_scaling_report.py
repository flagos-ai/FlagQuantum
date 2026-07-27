import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path("benchmarks/statevector_scaling_report.py")
SPEC = importlib.util.spec_from_file_location("statevector_scaling_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _artifact(world_size: int, samples: list[float]) -> dict:
    return {
        "schema_version": MODULE.SOURCE_SCHEMA,
        "artifact_class": "measured_development_run",
        "backend": "nccl",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "world_size": world_size,
        "workload_sha256": "same",
        "workload": {"n_wires": 24},
        "timing": {"samples_seconds": samples},
        "correctness": {
            "passed": True,
            "absolute_tolerance": 1e-5,
            "global_invariants": {"norm": 1.0, "z_expectations": {"0": 0.25}},
        },
        "rank_timings": [{"rank": rank} for rank in range(world_size)],
        "rank_peak_memory_bytes": [100] * world_size,
        "communication_fraction": 0.0,
        "hardware_inventory": {"cuda_device_names": ["A800"] * world_size},
    }


def test_report_computes_deterministic_bootstrap_speedup():
    report = MODULE.build_report(
        [_artifact(1, [10.0, 10.1, 9.9]), _artifact(2, [6.0, 6.1, 5.9])]
    )

    point = report["points"][1]
    assert point["speedup"]["speedup"] == pytest.approx(10.0 / 6.0)
    assert point["speedup"]["confidence_interval"][0] > 1.0
    assert report["scalability_claim_allowed"] is False
    assert report["release_gate_allowed"] is False


def test_report_rejects_mismatched_workloads_and_failed_correctness():
    candidate = _artifact(2, [6.0, 6.1, 5.9])
    candidate["workload_sha256"] = "different"
    with pytest.raises(ValueError, match="same workload_sha256"):
        MODULE.build_report([_artifact(1, [10.0, 10.1, 9.9]), candidate])

    candidate["workload_sha256"] = "same"
    candidate["correctness"] = {"passed": False}
    with pytest.raises(ValueError, match="pass correctness"):
        MODULE.build_report([_artifact(1, [10.0, 10.1, 9.9]), candidate])


def test_report_rejects_claimable_or_cross_world_inconsistent_sources():
    baseline = _artifact(1, [10.0, 10.1, 9.9])
    candidate = _artifact(2, [6.0, 6.1, 5.9])
    candidate["scalability_claim_allowed"] = True
    with pytest.raises(ValueError, match="reject scalability"):
        MODULE.build_report([baseline, candidate])

    candidate["scalability_claim_allowed"] = False
    candidate["correctness"]["global_invariants"]["z_expectations"]["0"] = 0.5
    with pytest.raises(ValueError, match="observables differ"):
        MODULE.build_report([baseline, candidate])


def test_report_rejects_multinode_source_without_topology_evidence():
    baseline = _artifact(1, [10.0, 10.1, 9.9])
    candidate = _artifact(2, [6.0, 6.1, 5.9])
    candidate["node_count"] = 2
    candidate["local_world_size"] = 1

    with pytest.raises(ValueError, match="rank placement"):
        MODULE.build_report([baseline, candidate])

    candidate["rank_placement"] = [
        {"rank": 0, "hostname": "node-a"},
        {"rank": 1, "hostname": "node-b"},
    ]
    with pytest.raises(ValueError, match="traffic evidence"):
        MODULE.build_report([baseline, candidate])
