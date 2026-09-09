"""Tests for native FlagQuantum backend capabilities."""

import pytest
import torch

import flagquantum as fq
import flagquantum.algorithms as algorithms
import flagquantum.compiler as compiler
import flagquantum.noise as fqn
import flagquantum.noise as noise
import flagquantum.runtime as fqr
import flagquantum.runtime.execution as execution
from flagquantum.algorithms import Hamiltonian
from flagquantum.compiler import CouplingMap
from flagquantum.runtime.backend_registry import (
    BackendCapabilities,
    backend_execution_options,
    capability_summary,
    get_backend_capabilities,
    refresh_backend_registry,
    register_backend,
    resolve_dtype,
)
from flagquantum.runtime.configuration import get_backend, set_backend, set_dtype
from flagquantum.runtime.planner import estimate_state_bytes, plan_for_backend

pytestmark = pytest.mark.integration


def test_default_backend_capabilities_are_pytorch_native():
    capabilities = get_backend_capabilities()

    assert capabilities.name == "pytorch"
    assert capabilities.tensor_backend == "torch"
    assert capabilities.supports_autograd
    assert capabilities.supports_statevector
    assert capabilities.supports_density_matrix
    assert capabilities.supports_mps
    assert capabilities.supports_distributed
    assert capabilities.supports_mode("adaptive_mps")
    assert capabilities.supports_mode("noisy_mps")


def test_backend_summary_is_serializable():
    summary = capability_summary()

    assert summary["name"] == "pytorch"
    assert "cpu" in summary["devices"]
    assert "complex64" in summary["dtypes"]
    assert isinstance(summary["accelerators"], tuple)


def test_runtime_backend_uses_registry():
    assert set_backend("torch") == "pytorch"
    assert get_backend() == "pytorch"


def test_resolve_device_auto_falls_back_to_available_device():
    device = fqr.resolve_device("auto")

    assert isinstance(device, torch.device)
    assert device.type in get_backend_capabilities().devices


def test_resolve_dtype_returns_real_and_complex_pair():
    real, complex_ = resolve_dtype("float64")

    assert real is torch.float64
    assert complex_ is torch.complex128
    assert set_dtype("complex64") == ("complex64", "float32")


def test_backend_execution_options_normalize_policy():
    options = backend_execution_options(mode="mps", device="auto", dtype="complex64")

    assert options["backend"] == "pytorch"
    assert options["mode"] == "mps"
    assert options["complex_dtype"] is torch.complex64
    assert options["device"] in get_backend_capabilities().devices


def test_register_backend_for_future_adapter_policy():
    capabilities = BackendCapabilities(
        name="example_adapter",
        tensor_backend="torch",
        devices=("cpu",),
        dtypes=("complex64",),
        supports_autograd=True,
        supports_distributed=False,
        supports_statevector=True,
        supports_density_matrix=False,
        supports_mps=False,
    )

    register_backend(capabilities, "example")

    assert get_backend_capabilities("example") == capabilities
    assert not get_backend_capabilities("example").supports_mode("distributed")
    refresh_backend_registry()


def test_run_native_accepts_auto_device_policy():
    circuit = fq.Circuit(1)
    circuit.h(0)

    state = fqr.run_native(circuit, device="auto")

    assert torch.allclose(state, circuit.state(), atol=1e-6)


def test_plan_for_backend_uses_dtype_and_topology_policy():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)

    plan = plan_for_backend(
        circuit,
        dtype="complex128",
        coupling_map=CouplingMap.line(3),
    )

    assert plan.state_bytes == estimate_state_bytes(3, complex_bytes=16)
    assert plan.analysis.gate_counts["swap"] == 2


def test_top_level_subsystems_remain_easy_to_use():
    assert algorithms.Hamiltonian is Hamiltonian
    assert compiler.CouplingMap is CouplingMap
    assert execution.run_native is fqr.run_native
    assert noise.NoiseModel is fqn.NoiseModel
