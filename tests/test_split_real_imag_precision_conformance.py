from __future__ import annotations

import os

import pytest

import flagquantum as fq


@pytest.mark.integration
def test_split_real_imag_precision_cpu_conformance() -> None:
    report = fq.experimental.run_split_real_imag_precision_conformance(
        "cpu", depths=(8, 32), seeds=(0, 7)
    )
    report.require_accepted()
    assert report.operator_profile == "split_real_imag_statevector_p2_precision"
    assert all(case.passed for case in report.cases)


@pytest.mark.integration
@pytest.mark.gpu
def test_split_real_imag_precision_cuda_conformance() -> None:
    if os.environ.get("FLAGQUANTUM_TEST_SPLIT_PRECISION_CUDA") != "1":
        pytest.skip("set FLAGQUANTUM_TEST_SPLIT_PRECISION_CUDA=1 on a CUDA runner")
    report = fq.experimental.run_split_real_imag_precision_conformance("cuda:0")
    report.require_accepted()
