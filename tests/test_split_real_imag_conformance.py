from __future__ import annotations

import pytest
import torch

from flagquantum.runtime.executors.statevector.split_real_imag import (
    run_split_real_imag_conformance,
)

pytestmark = pytest.mark.integration


def test_split_real_imag_cpu_conformance() -> None:
    report = run_split_real_imag_conformance("cpu")

    report.require_accepted()
    assert report.passed
    assert report.hardware_certification is False
    assert report.provider == "pytorch_cpu"
    assert tuple(case.depth for case in report.cases) == (8, 32, 128)
    assert all(case.passed for case in report.cases)


@pytest.mark.gpu
@pytest.mark.distributed_accel
def test_split_real_imag_cuda_conformance() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    report = run_split_real_imag_conformance("cuda:0")

    report.require_accepted()
    assert report.passed
    assert report.provider == "pytorch_cuda"
    assert all(case.passed for case in report.cases)
