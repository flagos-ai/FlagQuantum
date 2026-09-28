"""Statevector accelerated-kernel dispatch policy contracts."""

import importlib.util

import pytest

import flagquantum.runtime.executors.statevector.kernel_dispatch as kernel_dispatch
from flagquantum.runtime.executors.statevector.kernel_dispatch import (
    KernelDispatchEvidence,
    select_triton_kernel,
    triton_available,
)

pytestmark = pytest.mark.unit


def test_triton_availability_probe_is_cached(monkeypatch):
    calls = 0

    def available(name):
        nonlocal calls
        calls += 1
        assert name == "triton"
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", available)
    triton_available.cache_clear()
    try:
        assert triton_available()
        assert triton_available()
        assert calls == 1
    finally:
        triton_available.cache_clear()


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


def test_dispatch_evidence_aggregates_actual_decisions(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    evidence = KernelDispatchEvidence()

    evidence.record(
        select_triton_kernel("local_1q", requested=True, available=True), count=3
    )
    evidence.record(
        select_triton_kernel("local_cx", requested=True, available=False), count=2
    )

    summary = evidence.summary()
    assert summary["triton_execution_count"] == 3
    assert summary["pytorch_fallback_count"] == 2


def test_dispatch_records_direct_triton_compiler_identity(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    monkeypatch.setattr(
        kernel_dispatch,
        "triton_distribution_identity",
        lambda: ("triton", "3.7.1"),
    )

    decision = select_triton_kernel(
        "local_1q",
        requested=True,
        supported=True,
        available=True,
        device_runtime_provider="pytorch",
        device_type="cuda",
        compiler_backend="cuda",
        integration_path="direct",
    )

    assert decision.summary() == {
        "feature": "local_1q",
        "selected": "triton",
        "accelerated": True,
        "reason": "eligible",
        "device_runtime": {"provider": "pytorch", "device_type": "cuda"},
        "kernel_compiler": {
            "distribution": "triton",
            "version": "3.7.1",
            "backend": "cuda",
            "identity_source": "python_package_metadata",
        },
        "kernel_route": {
            "semantic_id": "statevector.local_1q",
            "implementation": "triton",
            "integration_path": "direct",
            "fallback": False,
        },
    }


def test_dispatch_records_direct_pytorch_fallback_without_compiler_claim(monkeypatch):
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")

    decision = select_triton_kernel(
        "local_1q",
        requested=True,
        supported=False,
        available=True,
        device_runtime_provider="pytorch",
        device_type="cpu",
        compiler_backend="cuda",
        integration_path="direct",
    )

    summary = decision.summary()
    assert summary["kernel_compiler"] is None
    assert summary["kernel_route"] == {
        "semantic_id": "statevector.local_1q",
        "implementation": "pytorch_eager",
        "integration_path": "direct",
        "fallback": True,
    }
