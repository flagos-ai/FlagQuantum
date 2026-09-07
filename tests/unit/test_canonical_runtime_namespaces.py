"""Canonical runtime namespace and removal guards."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.unit


def test_removed_runtime_stack_is_not_importable() -> None:
    assert importlib.util.find_spec("flagquantum.runtime_stack") is None


def test_removed_core_circuit_shim_is_not_importable() -> None:
    assert importlib.util.find_spec("flagquantum.core.circuit") is None


def test_public_circuit_uses_canonical_implementation() -> None:
    import flagquantum as fq
    from flagquantum.circuit import Circuit

    assert fq.Circuit is Circuit


def test_canonical_runtime_namespaces_are_importable() -> None:
    from flagquantum.runtime import audit, execution, training, training_state
    from flagquantum.runtime.backends import jax, mps, statevector, tensor_network
    from flagquantum.runtime.distributed import protocols

    assert audit is not None
    assert execution is not None
    assert training is not None
    assert training_state is not None
    assert protocols is not None
    assert jax is not None
    assert mps is not None
    assert statevector is not None
    assert tensor_network is not None


def test_canonical_runtime_does_not_reference_removed_namespace() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "flagquantum" / "runtime"
    for path in root.rglob("*.py"):
        assert "runtime_stack" not in path.read_text(encoding="utf-8"), path


def test_tensor_network_execution_is_owned_by_its_backend() -> None:
    from flagquantum.runtime import distributed
    from flagquantum.runtime.backends import tensor_network

    assert callable(tensor_network.run_distributed_tensor_network)
    assert not hasattr(distributed, "run_distributed_tensor_network")
