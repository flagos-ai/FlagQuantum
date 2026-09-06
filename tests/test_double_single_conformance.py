from __future__ import annotations

import json

import pytest

from flagquantum.numerics.conformance import _case, run_double_single_conformance

pytestmark = pytest.mark.integration


def test_cpu_double_single_conformance_is_machine_readable_and_passes() -> None:
    report = run_double_single_conformance("cpu")
    assert report.schema == "flagquantum_double_single_conformance_v1"
    assert report.compute_dtype == "float32"
    assert report.reference == "cpu_float64_and_complex128"
    assert report.runtime_integration_certified is False
    assert report.passed
    assert all(case.passed for case in report.cases)
    assert all(case.improvement_factor >= 1.0 for case in report.cases)
    assert json.loads(report.to_json())["passed"] is True


def test_conformance_allows_no_regression_when_fp32_is_already_exact() -> None:
    result = _case(
        "already_exact",
        1.0,
        1.0,
        1.0,
        {"already_exact": {"max_absolute_error": 0.0, "min_improvement_factor": 1e6}},
    )
    assert result.float32_baseline_within_threshold
    assert result.improvement_factor == 1.0
    assert result.passed


@pytest.mark.gpu
@pytest.mark.distributed_accel
def test_cuda_double_single_conformance_preserves_fp32_accuracy() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for accelerator-backed conformance")
    report = run_double_single_conformance("cuda:0")
    assert report.passed, report.to_dict()
    assert report.device == "cuda:0"
