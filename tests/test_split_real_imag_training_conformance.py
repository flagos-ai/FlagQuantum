from __future__ import annotations

import os

import pytest

from flagquantum.runtime.executors.statevector.split_real_imag import (
    run_split_real_imag_training_conformance,
)


@pytest.mark.integration
def test_split_real_imag_training_cpu_conformance() -> None:
    report = run_split_real_imag_training_conformance(
        "cpu", depths=(8, 32), seeds=(0, 7)
    )
    report.require_accepted()
    assert report.operator_profile == "split_real_imag_statevector_p1"
    assert all(case.passed for case in report.cases)


@pytest.mark.integration
@pytest.mark.gpu
def test_split_real_imag_training_cuda_conformance() -> None:
    if os.environ.get("FLAGQUANTUM_TEST_SPLIT_CUDA") != "1":
        pytest.skip("set FLAGQUANTUM_TEST_SPLIT_CUDA=1 on a CUDA runner")
    report = run_split_real_imag_training_conformance("cuda:0")
    report.require_accepted()
