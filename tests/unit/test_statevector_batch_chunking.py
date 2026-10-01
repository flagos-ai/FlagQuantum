"""CPU statevector batch windows preserve the public numerical contract."""

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.batching as statevector_batching
from flagquantum.errors import ValidationError

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
    actual = circuit.state(refresh=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert circuit._last_statevector_runtime["statevector_batch_chunk_size"] == 2
    assert circuit._last_statevector_runtime["statevector_batch_chunk_count"] == 3


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
