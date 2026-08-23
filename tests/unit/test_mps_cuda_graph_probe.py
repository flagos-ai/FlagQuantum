from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "internal"
    / "evidence"
    / "mps_cuda_graph_probe.py"
)
SPEC = importlib.util.spec_from_file_location("mps_cuda_graph_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cuda_graph_promotion_fails_closed_without_full_factorization_capture() -> None:
    result = MODULE._promotion(
        [
            {
                "speedup_compiled_over_graph": 2.0,
                "relative_error": 0.0,
            }
        ],
        minimum_speedup=1.05,
    )

    assert result["contraction_threshold_passed"]
    assert not result["full_factorization_capture_supported"]
    assert not result["allowed"]
    assert result["blockers"] == ["cusolver_gesvd_invalidates_cuda_graph_capture"]


def test_cuda_graph_promotion_records_contraction_threshold_failure() -> None:
    result = MODULE._promotion(
        [
            {
                "speedup_compiled_over_graph": 1.01,
                "relative_error": 0.0,
            }
        ],
        minimum_speedup=1.05,
    )

    assert not result["contraction_threshold_passed"]
    assert not result["allowed"]
