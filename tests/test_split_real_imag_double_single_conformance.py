from __future__ import annotations

import os

import pytest

from flagquantum.runtime.backends.statevector.split_real_imag_double_single_conformance import (
    run_split_real_imag_double_single_conformance,
)


@pytest.mark.integration
def test_split_real_imag_double_single_cpu_conformance() -> None:
    report = run_split_real_imag_double_single_conformance(
        "cpu", depths=(8, 32), seeds=(0, 7)
    )
    report.require_accepted()
    assert report.operator_profile == "split_real_imag_statevector_p3_double_single"
    assert all(case.passed for case in report.cases)
    assert report.host_gate_encoding is True
    assert report.state_host_fallback is False


@pytest.mark.integration
@pytest.mark.gpu
def test_split_real_imag_double_single_cuda_conformance() -> None:
    if os.environ.get("FLAGQUANTUM_TEST_SPLIT_DOUBLE_SINGLE_CUDA") != "1":
        pytest.skip("set FLAGQUANTUM_TEST_SPLIT_DOUBLE_SINGLE_CUDA=1 on a CUDA runner")
    report = run_split_real_imag_double_single_conformance("cuda:0")
    report.require_accepted()
