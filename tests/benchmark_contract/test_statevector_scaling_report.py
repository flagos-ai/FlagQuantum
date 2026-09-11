import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path("flagquantum/benchmarking/statevector_scaling_report.py")
SPEC = importlib.util.spec_from_file_location("statevector_scaling_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _artifact(world_size: int, samples: list[float]) -> dict[str, Any]:
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


def test_report_computes_deterministic_bootstrap_speedup() -> None:
    report = MODULE.build_report(
        [_artifact(1, [10.0, 10.1, 9.9]), _artifact(2, [6.0, 6.1, 5.9])]
    )

    point = report["points"][1]
    assert point["speedup"]["speedup"] == pytest.approx(10.0 / 6.0)
    assert point["speedup"]["confidence_interval"][0] > 1.0
    assert report["scalability_claim_allowed"] is False
    assert report["release_gate_allowed"] is False


def test_report_rejects_mismatched_workloads_and_failed_correctness() -> None:
    candidate = _artifact(2, [6.0, 6.1, 5.9])
    candidate["workload_sha256"] = "different"
    with pytest.raises(ValueError, match="same workload_sha256"):
        MODULE.build_report([_artifact(1, [10.0, 10.1, 9.9]), candidate])

    candidate["workload_sha256"] = "same"
    candidate["correctness"] = {"passed": False}
    with pytest.raises(ValueError, match="pass correctness"):
        MODULE.build_report([_artifact(1, [10.0, 10.1, 9.9]), candidate])


def test_report_rejects_claimable_or_cross_world_inconsistent_sources() -> None:
    baseline = _artifact(1, [10.0, 10.1, 9.9])
    candidate = _artifact(2, [6.0, 6.1, 5.9])
    candidate["scalability_claim_allowed"] = True
    with pytest.raises(ValueError, match="reject scalability"):
        MODULE.build_report([baseline, candidate])

    candidate["scalability_claim_allowed"] = False
    candidate["correctness"]["global_invariants"]["z_expectations"]["0"] = 0.5
    with pytest.raises(ValueError, match="observables differ"):
        MODULE.build_report([baseline, candidate])


def test_report_rejects_multinode_source_without_topology_evidence() -> None:
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


@pytest.mark.parametrize(
    "invalid", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("world_size", [1, 2])
def test_report_rejects_invalid_timing_samples(invalid: float, world_size: int) -> None:
    artifacts = [_artifact(1, [10.0, 10.1, 9.9])]
    if world_size == 2:
        artifacts.append(_artifact(2, [6.0, 6.1, 5.9]))
    artifacts[-1]["timing"]["samples_seconds"][0] = invalid

    with pytest.raises(ValueError, match="finite positive"):
        MODULE.build_report(artifacts)


@pytest.mark.parametrize(
    "invalid", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("source", ["baseline", "candidate"])
def test_bootstrap_rejects_invalid_timing_samples(invalid: float, source: str) -> None:
    samples = {"baseline": [10.0, 10.1, 9.9], "candidate": [6.0, 6.1, 5.9]}
    samples[source][0] = invalid

    with pytest.raises(ValueError, match="finite positive"):
        MODULE.bootstrap_median_speedup(**samples, resamples=1000)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("source_index", [0, 1])
@pytest.mark.parametrize("metric", ["norm", "z_expectations"])
def test_report_rejects_nonfinite_correctness_metrics(
    invalid: float, source_index: int, metric: str
) -> None:
    artifacts = [_artifact(1, [10.0, 10.1, 9.9]), _artifact(2, [6.0, 6.1, 5.9])]
    invariants = artifacts[source_index]["correctness"]["global_invariants"]
    if metric == "norm":
        invariants[metric] = invalid
    else:
        invariants[metric]["0"] = invalid

    with pytest.raises(ValueError, match="global (norm|observables)"):
        MODULE.build_report(artifacts)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("source_index", [0, 1])
def test_report_rejects_invalid_correctness_tolerance(
    invalid: float, source_index: int
) -> None:
    artifacts = [_artifact(1, [10.0, 10.1, 9.9]), _artifact(2, [6.0, 6.1, 5.9])]
    artifacts[source_index]["correctness"]["absolute_tolerance"] = invalid

    with pytest.raises(ValueError, match="finite nonnegative"):
        MODULE.build_report(artifacts)


@pytest.mark.parametrize("tolerance", [0.0, 0.125])
def test_report_accepts_correctness_at_tolerance_boundary(tolerance: float) -> None:
    baseline = _artifact(1, [10.0, 10.1, 9.9])
    candidate = _artifact(2, [6.0, 6.1, 5.9])
    for artifact in (baseline, candidate):
        artifact["correctness"]["absolute_tolerance"] = tolerance
    candidate["correctness"]["global_invariants"]["z_expectations"]["0"] += tolerance

    report = MODULE.build_report([baseline, candidate])

    assert report["points"][1]["speedup"]["speedup"] == pytest.approx(10.0 / 6.0)
