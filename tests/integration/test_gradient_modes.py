"""End-to-end evidence for the public gradient entry point.

`tests/test_gradient_api.py` states the contract of `fq.gradient` in isolation.
These tests execute real circuits through `fq.run` and check that the entry point
reports the method it actually used in every execution mode that can
differentiate, and that the modes agree with each other. They are integration
tests because they cross planning, runtime, simulation, and autograd.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.integration

DIFFERENTIABLE_MODES = ("statevector", "mps", "tensor_network")
PARAMETERS = torch.tensor([0.3, 0.7, -0.4], dtype=torch.float64)


def _circuit(parameters: torch.Tensor) -> fq.Circuit:
    return (
        fq.Circuit(2)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )


def _loss_in(mode: str) -> Callable[[fq.Circuit], torch.Tensor]:
    def loss(circuit: fq.Circuit) -> torch.Tensor:
        result = fq.run(
            circuit,
            options=fq.ExecutionOptions(mode=mode),
            outputs=fq.expectation(fq.Z(0)),
        )
        return result.expectations[0]

    return loss


def _reference() -> torch.Tensor:
    return fq.gradient(_circuit, PARAMETERS, _loss_in("statevector")).gradient


@pytest.mark.parametrize("mode", DIFFERENTIABLE_MODES)
def test_every_differentiable_mode_runs_and_reports_the_method(mode: str) -> None:
    result = fq.gradient(_circuit, PARAMETERS, _loss_in(mode))

    assert result.method == "autograd"
    assert result.exact is True
    assert result.step is None
    assert result.gradient.shape == PARAMETERS.shape
    assert result.gradient.dtype == PARAMETERS.dtype


@pytest.mark.parametrize("mode", DIFFERENTIABLE_MODES)
def test_every_differentiable_mode_returns_the_same_derivative(mode: str) -> None:
    """The mode selects how amplitudes are stored, not what the derivative is."""

    result = fq.gradient(_circuit, PARAMETERS, _loss_in(mode))

    assert torch.allclose(result.gradient, _reference(), rtol=1e-9, atol=1e-12)


def test_the_single_qubit_derivative_matches_the_closed_form() -> None:
    """A route that agreed with itself could still be wrong, so check analysis."""

    def one_qubit(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1).ry(0, theta=parameters[0])

    def one_qubit_loss(circuit: fq.Circuit) -> torch.Tensor:
        return fq.run(circuit, outputs=fq.expectation(fq.Z(0))).expectations[0]

    parameters = torch.tensor([0.3], dtype=torch.float64)
    result = fq.gradient(one_qubit, parameters, one_qubit_loss)

    # <Z> after Ry(t) on |0> is cos(t), whose derivative is -sin(t).
    assert torch.allclose(result.gradient, -torch.sin(parameters), rtol=1e-6, atol=1e-9)


def test_the_reference_derivative_matches_an_independent_difference() -> None:
    """The reference gradient is re-derived by finite differences of the loss."""

    reference = _reference()
    step = 1e-4
    independent = []
    for index in range(PARAMETERS.numel()):
        shifted = []
        for sign in (1.0, -1.0):
            moved = PARAMETERS.clone()
            moved[index] += sign * step
            shifted.append(float(_loss_in("statevector")(_circuit(moved))))
        independent.append((shifted[0] - shifted[1]) / (2.0 * step))

    assert torch.allclose(
        reference,
        torch.tensor(independent, dtype=reference.dtype),
        rtol=1e-2,
        atol=1e-4,
    )
    # Every entry moving is what makes the agreement above meaningful.
    assert float(reference.abs().min()) > 0.1


def test_the_result_carries_no_live_graph_into_a_second_step() -> None:
    """A training loop reuses the tensor, so it must not extend the tape."""

    result = fq.gradient(_circuit, PARAMETERS, _loss_in("statevector"))

    assert result.gradient.requires_grad is False
    assert result.gradient.grad_fn is None
    assert PARAMETERS.requires_grad is False


def test_the_shift_rule_agrees_with_the_mode_that_executed() -> None:
    """The exact rule is declared per opcode and is mode-independent."""

    exact = fq.gradient(_circuit, PARAMETERS, _loss_in("statevector"))
    shifted = fq.gradient(
        _circuit, PARAMETERS, _loss_in("mps"), method="parameter_shift"
    )

    assert shifted.method == "parameter_shift"
    assert shifted.exact is True
    assert torch.allclose(shifted.gradient, exact.gradient, rtol=1e-6, atol=1e-9)


def test_a_mode_that_cannot_serve_the_loss_fails_closed() -> None:
    """The stabilizer mode samples outcomes and cannot serve an expectation."""

    with pytest.raises(CapabilityError, match="stabilizer"):
        fq.gradient(_circuit, PARAMETERS, _loss_in("stabilizer"))
