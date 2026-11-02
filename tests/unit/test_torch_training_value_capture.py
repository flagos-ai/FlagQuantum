"""Canonical numerical capture keeps mutable frozen inputs and real gradients."""

import pytest
import torch

import flagquantum as fq
from flagquantum.core.parameters import Parameter
from flagquantum.errors import CapabilityError
from flagquantum.simulation.statevector.torch_training import capture_training_value

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("batch", (1, 8))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    "mask", ((False, False), (False, True), (True, False), (True, True))
)
def test_graph_preserves_current_values_nonuniform_vjps_and_no_grad(batch, dtype, mask):
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    program = (
        fq.Circuit(2, dtype=complex_dtype)
        .ry(0, Parameter("a"))
        .rx(1, Parameter("b"))
        .cx(0, 1)
        .to_ir()
    )
    initial = {
        name: torch.linspace(0.2, 0.4, batch, dtype=dtype).requires_grad_(grad)
        for name, grad in zip(("a", "b"), mask, strict=True)
    }
    graph = capture_training_value(
        program,
        initial,
        batch_size=batch,
        device=torch.device("cpu"),
        dtype=complex_dtype,
        observable_qubits=(0, 1),
        sum_observables=True,
    )
    tolerance = 1e-12 if dtype == torch.float64 else 2e-6
    for step in range(3):
        a = (initial["a"].detach() + step * 0.01).requires_grad_(mask[0])
        b = (initial["b"].detach() - step * 0.02).requires_grad_(mask[1])
        before = (a.detach().clone(), b.detach().clone())
        actual = graph(a, b)
        expected = a.cos() * (1 + b.cos())
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        selected = tuple(value for value in (a, b) if value.requires_grad)
        if selected:
            weights = torch.linspace(-0.7, 1.3, batch, dtype=dtype)
            observed = torch.autograd.grad((actual * weights).sum(), selected)
            reference = torch.autograd.grad((expected * weights).sum(), selected)
            torch.testing.assert_close(
                observed, reference, atol=tolerance, rtol=tolerance
            )
        torch.testing.assert_close((a, b), before, atol=0, rtol=0)
        with torch.no_grad():
            validation = graph(a, b)
            torch.testing.assert_close(
                validation, expected.detach(), atol=tolerance, rtol=tolerance
            )
            assert not validation.requires_grad


def test_capture_does_not_retain_the_original_example_storage():
    program = fq.Circuit(1, dtype=torch.complex128).ry(0, Parameter("a")).to_ir()
    example = torch.tensor([0.2], dtype=torch.float64)
    graph = capture_training_value(
        program,
        {"a": example},
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
        observable_qubits=(0,),
        sum_observables=False,
    )
    example.fill_(0.8)
    value = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(graph(value), value.cos(), atol=1e-12, rtol=1e-12)
    (first,) = torch.autograd.grad(graph(value).sum(), value, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), value)
    torch.testing.assert_close(second, -value.cos(), atol=1e-12, rtol=1e-12)


def test_explicit_matrices_and_unresolved_bindings_are_refused():
    program = (
        fq.Circuit(1)
        .ry(0, Parameter("a"))
        .gate("x", (0,), matrix=torch.eye(2, dtype=torch.complex128))
        .to_ir()
    )
    with pytest.raises(CapabilityError, match="plain gate IR"):
        capture_training_value(
            program,
            {"a": torch.tensor([0.2])},
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.complex128,
            observable_qubits=(0,),
            sum_observables=False,
        )
    plain = fq.Circuit(1).ry(0, Parameter("a")).to_ir()
    with pytest.raises(CapabilityError, match="every canonical binding"):
        capture_training_value(
            plain,
            {"other": torch.tensor([0.2])},
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.complex128,
            observable_qubits=(0,),
            sum_observables=False,
        )
