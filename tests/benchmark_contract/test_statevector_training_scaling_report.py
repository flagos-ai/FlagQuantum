from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("benchmarks/statevector_training_scaling_report.py")
sys.path.insert(0, str(SCRIPT.parent.resolve()))
SPEC = importlib.util.spec_from_file_location(
    "statevector_training_scaling_report", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _point(world: int) -> dict:
    phase = {
        "samples_seconds": [1.0, 1.01, 0.99],
        "median_seconds": 1.0,
        "coefficient_of_variation": 0.01,
    }
    return {
        "schema_version": "flagquantum.statevector.training_scaling.v1",
        "world_size": world,
        "node_count": 1,
        "rank_placement": [{"rank": rank} for rank in range(world)],
        "workload": {
            "n_wires": 10 + world.bit_length() - 1,
            "local_amplitudes": 1024,
            "dtype": "complex64",
        },
        "correctness": {"passed": True, "gradient_absolute_error_max": 1e-7},
        "forward": {**phase, "peak_memory_bytes_max": 2048},
        "differentiable_forward": dict(phase),
        "backward": dict(phase),
        "end_to_end": {**phase, "peak_memory_bytes_max": 4096},
        "backward_communication_bytes_per_rank_max": world * 100,
    }


def test_build_report_accepts_differentiable_weak_scaling() -> None:
    report = MODULE.build_report([_point(1), _point(2), _point(4)])
    assert report["source_world_sizes"] == [1, 2, 4]
    assert report["definition"] == "constant_2^30_amplitudes_per_rank"
    assert report["points"][-1]["n_wires"] == 12
    assert (
        report["points"][-1]["phases"]["backward"]["weak_scaling_efficiency"]["speedup"]
        == 1.0
    )


def test_build_report_rejects_gradient_failure() -> None:
    candidate = _point(2)
    candidate["correctness"]["passed"] = False
    try:
        MODULE.build_report([_point(1), candidate])
    except ValueError as error:
        assert "correctness" in str(error)
    else:
        raise AssertionError("failed gradients must reject the report")


def test_build_report_rejects_local_amplitude_drift() -> None:
    candidate = _point(2)
    candidate["workload"]["local_amplitudes"] = 2048
    try:
        MODULE.build_report([_point(1), candidate])
    except ValueError as error:
        assert "local amplitudes" in str(error)
    else:
        raise AssertionError("nonconstant local state must reject weak scaling")
