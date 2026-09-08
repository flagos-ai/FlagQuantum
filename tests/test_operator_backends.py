"""Tests for optional native PyTorch operator backends."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

import flagquantum as fq
import flagquantum.backends as fqb
from flagquantum.compute.flaggems import (
    flaggems_preflight,
    operator_backend,
    plan_operator_replacements,
    validate_flaggems_ops,
)


class _FakeUseGems:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def __init__(
        self, *, include=None, exclude=None, record=False, once=False, path=None
    ):
        self.include = tuple(include or ())

    def __enter__(self):
        type(self).calls.append(("enter", self.include))
        return self

    def __exit__(self, exc_type, exc, tb):
        type(self).calls.append(("exit", self.include))
        return False


def _install_fake_flaggems(monkeypatch, keys=("mm", "bmm", "sum", "einsum")):
    _FakeUseGems.calls.clear()
    fake = SimpleNamespace(
        __version__="test",
        device="cpu",
        vendor_name="test_vendor",
        use_gems=_FakeUseGems,
        all_registered_keys=lambda: tuple(keys),
    )
    monkeypatch.setitem(sys.modules, "flag_gems", fake)
    return fake


def _raise_current_registrar_error():
    raise AttributeError("'NoneType' object has no attribute 'get_all_keys'")


def test_flaggems_preflight_reports_unavailable_without_importable_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "flag_gems", None)

    report = flaggems_preflight(requested_ops=["mm", "einsum"])

    assert report["availability"]["available"] is False
    assert "mm" in report["replacement_plan"]["catalog_safe_ops"]
    assert "einsum" in report["replacement_plan"]["experimental_ops"]


def test_flaggems_preflight_uses_static_catalog_before_registered_keys(monkeypatch):
    def mm():
        return None

    def bmm():
        return None

    fake = SimpleNamespace(
        __version__="test",
        device="cuda",
        vendor_name="nvidia",
        FULL_CONFIG_BY_FUNC={"mm": [("mm", mm)]},
        _FULL_CONFIG=(("bmm", bmm),),
        use_gems=_FakeUseGems,
        all_registered_keys=_raise_current_registrar_error,
    )
    monkeypatch.setitem(sys.modules, "flag_gems", fake)

    report = flaggems_preflight(requested_ops=["mm", "bmm"])

    assert report["availability"]["available"] is True
    assert set(report["replacement_plan"]["runtime_replaceable_ops"]) == {"mm", "bmm"}


def test_flaggems_replacement_plan_classifies_runtime_and_experimental_ops(monkeypatch):
    _install_fake_flaggems(monkeypatch)

    plan = plan_operator_replacements(
        "flaggems",
        requested_ops=[
            "mm",
            "bmm",
            "einsum",
            "svd",
            "torch.distributed.all_to_all",
            "jax.pmap",
            "made_up_quantum_op",
        ],
    )

    assert plan.runtime_replaceable_ops == ("mm", "bmm")
    assert "einsum" in plan.experimental_ops
    assert "svd" in plan.experimental_ops
    assert "torch.distributed.all_to_all" in plan.non_operator_requirements
    assert "jax.pmap" in plan.non_operator_requirements
    assert "made_up_quantum_op" in plan.unavailable_ops


def test_operator_backend_normalizes_aten_overload_aliases(monkeypatch):
    _install_fake_flaggems(monkeypatch, keys=("where_self",))

    with operator_backend("flaggems", include=["where.self"], strict=True) as session:
        assert session.enabled

    assert _FakeUseGems.calls == [("enter", ("where_self",)), ("exit", ("where_self",))]


def test_validate_flaggems_ops_runs_complex_smoke(monkeypatch):
    _install_fake_flaggems(monkeypatch, keys=("mm", "mul"))

    result = validate_flaggems_ops(["mm", "mul"], device="cpu", dtype="complex64")

    assert result.passed_ops == ("mm", "mul")
    assert result.failed_ops == {}


def test_validate_flaggems_ops_reports_unimplemented_validator(monkeypatch):
    _install_fake_flaggems(monkeypatch, keys=("made_up",))

    result = validate_flaggems_ops(["made_up"], device="cpu", dtype="complex64")

    assert result.passed_ops == ()
    assert "made_up" in result.failed_ops


def test_operator_backend_context_enables_only_safe_requested_ops(monkeypatch):
    _install_fake_flaggems(monkeypatch, keys=("mm", "bmm", "einsum"))

    with operator_backend("flaggems", include=["mm", "einsum"], strict=True) as session:
        assert session.enabled

    assert _FakeUseGems.calls == [("enter", ("mm",)), ("exit", ("mm",))]


def test_run_native_accepts_flaggems_operator_backend(monkeypatch):
    _install_fake_flaggems(monkeypatch, keys=("mm", "bmm", "sum"))
    circuit = fq.Circuit(1)
    circuit.rx(0, theta=torch.tensor(0.2))

    state = fqb.run_native(
        circuit,
        mode="statevector",
        operator_backend="flaggems",
        operator_backend_include=["mm", "bmm", "sum"],
        operator_backend_strict=True,
    )

    assert torch.allclose(state, circuit.state(), atol=1e-6)
    assert _FakeUseGems.calls[0] == ("enter", ("mm", "bmm", "sum"))
    assert _FakeUseGems.calls[-1] == ("exit", ("mm", "bmm", "sum"))


def test_operator_backend_context_fails_open_when_flaggems_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "flag_gems", None)

    with operator_backend("flaggems", include=["mm"], strict=False) as session:
        assert not session.enabled
        assert not session.availability.available
