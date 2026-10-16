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


def _precise_circuit(parameters: torch.Tensor) -> fq.Circuit:
    """The same two-qubit circuit in double precision.

    The velocity tests differentiate twice, and a second-order route over the
    default single-precision amplitudes differs from a single-order one at the
    single-precision epsilon. Widening the circuit is what keeps the tolerance
    below a statement about the mode rather than about the dtype.
    """

    return (
        fq.Circuit(2, dtype=torch.complex128)
        .ry(0, theta=parameters[0])
        .rx(0, theta=parameters[1])
        .ry(1, theta=parameters[2])
        .cx(1, 0)
    )


def _vector_output_in(mode: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return one expectation value per qubit after a full two-qubit circuit."""

    def program(parameters: torch.Tensor) -> torch.Tensor:
        result = fq.run(
            _precise_circuit(parameters),
            options=fq.ExecutionOptions(mode=mode),
            outputs=[fq.expectation(fq.Z(0)), fq.expectation(fq.Z(1))],
        )
        return torch.stack(list(result.expectations))

    return program


def _reference_jacobian() -> torch.Tensor:
    return fq.jacobian(_vector_output_in("statevector"), PARAMETERS)


@pytest.mark.parametrize("mode", DIFFERENTIABLE_MODES)
def test_every_differentiable_mode_returns_the_same_jacobian(mode: str) -> None:
    """The mode selects how amplitudes are stored, not what the derivative is."""

    result = fq.jacobian(_vector_output_in(mode), PARAMETERS)

    assert result.shape == (*_vector_output_in(mode)(PARAMETERS).shape, 3)
    assert result.dtype == PARAMETERS.dtype
    assert torch.allclose(result, _reference_jacobian(), rtol=1e-12, atol=1e-15)
    # Four of the six entries move. The two zeros are structural: qubit 1 is
    # rotated only by the third parameter, so its expectation cannot depend on
    # the first two. Stating that keeps the agreement above from being an
    # agreement about an all-zero matrix.
    assert int((result.reshape(-1).abs() > 1e-3).sum()) == 4


@pytest.mark.parametrize("mode", DIFFERENTIABLE_MODES)
def test_every_differentiable_mode_returns_the_same_directional_products(
    mode: str,
) -> None:
    """Forward and reverse directions must agree across every mode."""

    output = _vector_output_in(mode)(PARAMETERS)
    tangent = torch.tensor([0.4, 1.1, -0.9], dtype=torch.float64)
    cotangent = torch.tensor([-0.6, 2.0], dtype=torch.float64).reshape(output.shape)
    reference = _reference_jacobian().reshape(output.numel(), 3)

    forward = fq.jvp(_vector_output_in(mode), PARAMETERS, tangent)
    reverse = fq.vjp(_vector_output_in(mode), PARAMETERS, cotangent)

    assert torch.allclose(
        forward.reshape(-1), reference @ tangent, rtol=1e-10, atol=1e-13
    )
    assert torch.allclose(
        reverse,
        (reference.T @ cotangent.reshape(-1)).reshape_as(PARAMETERS),
        rtol=1e-12,
        atol=1e-15,
    )


def test_a_training_step_shaped_objective_uses_the_reverse_product() -> None:
    """A scalar objective over several measured values is the reverse shape.

    The objective weights both expectation values, so one reverse sweep returns
    the whole parameter update. It must match the parameter gradient of the same
    scalar objective taken through the scalar entry point.
    """

    output = _vector_output_in("statevector")(PARAMETERS)
    cotangent = torch.tensor([1.5, -0.25], dtype=torch.float64).reshape(output.shape)

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        measured = fq.run(
            _precise_circuit(parameters),
            outputs=[fq.expectation(fq.Z(0)), fq.expectation(fq.Z(1))],
        )
        stacked = torch.stack(list(measured.expectations))
        return (stacked * cotangent).sum()

    reverse = fq.vjp(_vector_output_in("statevector"), PARAMETERS, cotangent)
    scalar = fq.gradient(objective, PARAMETERS)

    assert scalar.method == "autograd"
    assert torch.allclose(reverse, scalar.gradient, rtol=1e-12, atol=1e-15)
    # Both terms must contribute, or the agreement would be about one of them.
    assert float(reverse.abs().min()) > 1e-2


def test_a_mode_that_cannot_serve_the_jacobian_fails_closed() -> None:
    """The stabilizer mode samples outcomes and cannot serve an expectation."""

    with pytest.raises(CapabilityError, match="stabilizer"):
        fq.jacobian(_vector_output_in("stabilizer"), PARAMETERS)
