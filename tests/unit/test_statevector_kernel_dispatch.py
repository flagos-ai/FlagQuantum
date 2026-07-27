"""Statevector accelerated-kernel dispatch policy contracts."""

import pytest

from flagquantum.runtime.backends.statevector.kernel_dispatch import (
    select_triton_kernel,
)

pytestmark = pytest.mark.unit


def test_dispatch_selects_accelerated_kernel_when_eligible(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")

    decision = select_triton_kernel(
        "local_1q", requested=True, supported=True, available=True
    )

    assert decision.accelerated
    assert decision.summary() == {
        "feature": "local_1q",
        "selected": "triton",
        "accelerated": True,
        "reason": "eligible",
    }


@pytest.mark.parametrize(
    ("requested", "supported", "available", "reason"),
    (
        (False, True, True, "disabled_by_policy"),
        (True, True, False, "triton_unavailable"),
        (True, False, True, "input_not_supported"),
    ),
)
def test_dispatch_records_portable_fallback_reason(
    monkeypatch, requested, supported, available, reason
):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")

    decision = select_triton_kernel(
        "local_cx",
        requested=requested,
        supported=supported,
        available=available,
    )

    assert not decision.accelerated
    assert decision.selected == "pytorch"
    assert decision.reason == reason


def test_portable_mode_has_precedence_over_accelerator_availability(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "portable")

    decision = select_triton_kernel(
        "vjp_adjoint", requested=True, supported=True, available=True
    )

    assert not decision.accelerated
    assert decision.reason == "portable_mode"
