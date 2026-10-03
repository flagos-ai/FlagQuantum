"""CPU statevector batch windows preserve the public numerical contract."""

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.batching as statevector_batching
from flagquantum.core.ir import Instruction
from flagquantum.errors import ValidationError
from flagquantum.simulation.statevector.program import (
    _preallocated_batch_assembly_beneficial,
    _StatevectorCliffordMatchingStep,
    _StatevectorFusedGateStep,
)

pytestmark = pytest.mark.unit


def _circuit(theta: torch.Tensor) -> fq.Circuit:
    return (
        fq.Circuit(3, bsz=theta.shape[0], dtype=torch.complex128)
        .ry(0, theta)
        .cx(0, 1)
        .rz(2, theta * 0.7)
        .h(1)
        .cx(2, 0)
    )


def test_cpu_batch_chunk_size_respects_logical_state_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = torch.empty((5, 16), dtype=torch.complex128)
    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        2 * 16 * state.element_size(),
    )

    assert statevector_batching._cpu_statevector_batch_chunk_size(state) == 2
    monkeypatch.setenv("FQ_CPU_STATEVECTOR_BATCH_CHUNKING", "0")
    assert statevector_batching._cpu_statevector_batch_chunk_size(state) == 5


def test_preallocated_assembly_avoids_mixed_rotation_clifford_workspace() -> None:
    rotation = _StatevectorFusedGateStep(
        instructions=(
            Instruction("ry", (0,), {"theta": 0.1}),
            Instruction("rz", (0,), {"theta": 0.2}),
        ),
        wires=(0,),
        layout=((0,), (1,)),
    )
    matching = _StatevectorCliffordMatchingStep(
        controls=(0,), targets=(1,), cz_edges=()
    )

    assert _preallocated_batch_assembly_beneficial((rotation,)) is True
    assert _preallocated_batch_assembly_beneficial((matching,)) is True
    assert _preallocated_batch_assembly_beneficial((rotation, matching)) is False


def test_chunked_batch_matches_monolithic_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = torch.tensor([0.1, -0.4, 0.7, 1.1, -0.9], dtype=torch.float64)
    circuit = _circuit(theta)
    monkeypatch.setenv("FQ_CPU_STATEVECTOR_BATCH_CHUNKING", "0")
    expected = circuit.state(refresh=True)

    monkeypatch.setenv("FQ_CPU_STATEVECTOR_BATCH_CHUNKING", "1")
    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        2 * (2**circuit.n_wires) * torch.empty((), dtype=circuit.dtype).element_size(),
    )
    chunked = _circuit(theta)
    actual = chunked.state(refresh=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert chunked._last_statevector_runtime["statevector_batch_chunk_size"] == 2
    assert chunked._last_statevector_runtime["statevector_batch_chunk_count"] == 3
    assert chunked._initial_state_workspace is None
    assert chunked._initial_state_batch_window_workspace is not None
    assert chunked._initial_state_batch_window_workspace.shape == (2, 2**3)


def test_chunk_length_does_not_legalize_invalid_full_batch_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = fq.Circuit(3, bsz=5, dtype=torch.complex128).ry(
        0, torch.tensor([0.1, 0.2], dtype=torch.float64)
    )
    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        2 * (2**circuit.n_wires) * torch.empty((), dtype=circuit.dtype).element_size(),
    )

    with pytest.raises(ValidationError, match="expected 1 or the batch size 5"):
        circuit.state()


def test_chunked_batch_preserves_parameter_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = torch.tensor([0.2, -0.3, 0.8, -1.0], dtype=torch.float64)

    monolithic_theta = values.clone().requires_grad_(True)
    monkeypatch.setenv("FQ_CPU_STATEVECTOR_BATCH_CHUNKING", "0")
    monolithic_state = _circuit(monolithic_theta).state()
    monolithic_loss = monolithic_state.real.square().sum()
    monolithic_loss.backward()

    chunked_theta = values.clone().requires_grad_(True)
    monkeypatch.setenv("FQ_CPU_STATEVECTOR_BATCH_CHUNKING", "1")
    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        (2**3) * torch.empty((), dtype=torch.complex128).element_size(),
    )
    chunked_state = _circuit(chunked_theta).state()
    chunked_loss = chunked_state.real.square().sum()
    chunked_loss.backward()

    torch.testing.assert_close(chunked_state, monolithic_state, rtol=0, atol=0)
    torch.testing.assert_close(
        chunked_theta.grad, monolithic_theta.grad, rtol=1e-12, atol=1e-12
    )


def test_chunked_inference_bounds_the_zero_state_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = torch.tensor([0.1, -0.4, 0.7, 1.1, -0.9], dtype=torch.float64)
    circuit = _circuit(theta)
    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        2 * (2**circuit.n_wires) * torch.empty((), dtype=circuit.dtype).element_size(),
    )
    actual = circuit.state()

    expected = _circuit(theta).state()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (
        circuit._last_statevector_runtime["statevector_batch_assembly"]
        == "preallocated_copy"
    )


def test_chunked_inference_preallocated_assembly_has_a_complete_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = torch.tensor([0.1, -0.4, 0.7, 1.1, -0.9], dtype=torch.float64)
    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        2 * (2**3) * torch.empty((), dtype=torch.complex128).element_size(),
    )
    monkeypatch.setenv("FQ_CPU_STATEVECTOR_BATCH_PREALLOCATED_ASSEMBLY", "0")
    circuit = _circuit(theta)

    result = circuit.state()

    torch.testing.assert_close(result, _circuit(theta).state(), rtol=0, atol=0)
    assert (
        circuit._last_statevector_runtime["statevector_batch_assembly"]
        == "functional_cat"
    )


def test_chunked_bounded_initial_state_has_a_complete_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = torch.tensor([0.1, -0.4, 0.7, 1.1, -0.9], dtype=torch.float64)
    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        2 * (2**3) * torch.empty((), dtype=torch.complex128).element_size(),
    )
    monkeypatch.setenv("FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE", "0")
    circuit = _circuit(theta)

    result = circuit.state()

    assert result.shape == (5, 2**3)
    assert circuit._initial_state_workspace is not None
    assert circuit._initial_state_batch_window_workspace is None
    assert (
        circuit._last_statevector_runtime["statevector_batch_assembly"]
        == "preallocated_copy"
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_chunked_native_layer_initializes_final_slices_directly(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    def circuit() -> fq.Circuit:
        result = fq.Circuit(4, bsz=5, dtype=dtype)
        for layer in range(2):
            for wire in reversed(range(4)):
                result.ry(wire, 0.1 * (layer + 1) * (wire + 1))
                result.rz(wire, -0.07 * (layer + 1) * (wire + 1))
            result.cx(3, 2).cx(1, 0)
        return result

    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        2 * (2**4) * torch.empty((), dtype=dtype).element_size(),
    )
    selected = circuit()
    actual = selected.state()
    monkeypatch.setenv("FQ_CPU_NATIVE_PRODUCT_STATE_INITIALIZATION", "0")
    initializer_rollback = circuit()
    fallback = initializer_rollback.state()
    monkeypatch.delenv("FQ_CPU_NATIVE_PRODUCT_STATE_INITIALIZATION")
    monkeypatch.setenv("FQ_CPU_STATEVECTOR_BATCH_DIRECT_ASSEMBLY", "0")
    rollback = circuit()
    expected = rollback.state()

    tolerance = 1e-6 if dtype == torch.complex64 else 1e-14
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(fallback, expected, rtol=tolerance, atol=tolerance)
    assert selected._initial_state_batch_window_workspace is None
    assert selected._last_statevector_runtime["statevector_batch_assembly"] == (
        "direct_preallocated"
    )
    assert (
        selected._last_statevector_runtime["native_cpu_product_state_initialization"]
        == 1
    )
    assert (
        initializer_rollback._last_statevector_runtime["statevector_batch_assembly"]
        == "direct_preallocated"
    )
    assert (
        initializer_rollback._last_statevector_runtime[
            "native_cpu_product_state_initialization"
        ]
        == 0
    )
    assert rollback._last_statevector_runtime["statevector_batch_assembly"] == (
        "preallocated_copy"
    )


def test_chunked_batch_slices_compiled_module_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        statevector_batching,
        "_CPU_STATEVECTOR_BATCH_CHUNK_BUDGET_BYTES",
        (2**2) * torch.empty((), dtype=torch.complex64).element_size(),
    )

    def builder(parameters: torch.Tensor, inputs: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(2, bsz=inputs.shape[0]).ry(0, inputs + parameters[0]).cx(0, 1)

    module = fq.Module(
        builder,
        1,
        policy=fq.RuntimePolicy(observable="z", observable_qubits=(0,)),
    )
    inputs = torch.tensor([0.1, 0.3, -0.8], requires_grad=True)
    value = module(inputs)
    value.sum().backward()

    expected = torch.cos(inputs.detach() + module.parameters_tensor.detach()[0])
    torch.testing.assert_close(value, expected)
    assert inputs.grad is not None
    assert module.parameters_tensor.grad is not None
