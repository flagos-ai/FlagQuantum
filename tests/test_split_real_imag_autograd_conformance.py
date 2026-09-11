from __future__ import annotations

import pytest

from flagquantum.runtime.executors.statevector.split_real_imag_autograd_conformance import (
    run_split_real_imag_autograd_conformance,
)

pytestmark = pytest.mark.integration


def test_split_real_imag_p5_autograd_cpu_conformance() -> None:
    report = run_split_real_imag_autograd_conformance("cpu")
    report.require_accepted()
    assert report.passed is True
    assert report.provider == "pytorch_cpu"
    assert report.delivered_tensor_grad_precision == "float32_boundary"
    assert report.internal_gradient_representation == "double_single_high_low"
    assert report.optimizer_available is True
    assert report.optimizer_evidence_in_report is False
    assert report.native_cuda_evidence is False
    assert report.torch_fl_flagos_evidence is False
