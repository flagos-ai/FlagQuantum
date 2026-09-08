from __future__ import annotations

import pytest

from flagquantum.runtime.executors.statevector.split_real_imag_optimizer_conformance import (
    run_split_real_imag_optimizer_conformance,
)

pytestmark = pytest.mark.integration


def test_split_real_imag_p5_optimizer_cpu_trajectory_conformance() -> None:
    report = run_split_real_imag_optimizer_conformance("cpu")
    report.require_accepted()
    assert report.passed is True
    assert report.tensor_grad_used is False
    assert report.cancellation_double_single_absolute_error == 0.0
    assert report.cancellation_float32_absolute_error > 0.0
    assert all(
        case.double_single_parameter_absolute_error
        <= case.float32_parameter_absolute_error
        for case in report.cases
    )
    assert all(
        case.double_single_loss_absolute_error <= case.float32_loss_absolute_error
        for case in report.cases
    )
    assert report.native_cuda_evidence is False
    assert report.torch_fl_flagos_evidence is False
    assert report.convergence_certification is False
