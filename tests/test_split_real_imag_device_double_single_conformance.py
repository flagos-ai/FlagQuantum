from __future__ import annotations

import pytest

from flagquantum.runtime.executors.statevector.split_real_imag_device_double_single_conformance import (
    run_split_real_imag_device_double_single_conformance,
)

pytestmark = pytest.mark.integration


def test_split_p4_device_gate_conformance_cpu() -> None:
    report = run_split_real_imag_device_double_single_conformance(
        "cpu",
        depths=(2,),
        seeds=(0,),
        stability_depths=(8,),
        stability_seeds=(0,),
    )
    report.require_accepted()
    assert report.device_only_double_single_trigonometry is True
    assert report.host_gate_encoding is False
    assert report.parameter_host_fallback is False
    assert report.state_host_fallback is False
