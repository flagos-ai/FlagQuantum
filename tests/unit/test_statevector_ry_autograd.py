"""Protect analytic RY gradients for complex states and batched angles."""

import pytest
import torch

from flagquantum.simulation.statevector.operations import _apply_matrix_layout
from flagquantum.simulation.statevector.rotation_autograd import _RYRotation

pytestmark = pytest.mark.unit


def _matrix(parameters: torch.Tensor) -> torch.Tensor:
    half_angle = parameters[:, 0] * 0.5
    cosine, sine = half_angle.cos(), half_angle.sin()
    values = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(-1, 2, 2)
    return torch.complex(values, torch.zeros_like(values))


@pytest.mark.parametrize("batch,shared", [(1, True), (2, True), (2, False)])
@pytest.mark.parametrize("wire", [0, 1, 3, 5])
def test_ry_complex_first_and_second_derivatives(
    batch: int, shared: bool, wire: int
) -> None:
    torch.manual_seed(713 + wire)
    state = torch.randn(batch, 64, dtype=torch.complex128, requires_grad=True)
    parameters = torch.randn(
        1 if shared else batch, 1, dtype=torch.float64, requires_grad=True
    )
    matrix = _matrix(parameters)
    reference = _apply_matrix_layout(state, matrix.expand(batch, -1, -1), (wire,), 6)
    measured = _RYRotation.apply(state, parameters, matrix.detach(), wire, 6)
    weight = torch.randn_like(state)
    torch.testing.assert_close(measured, reference, atol=1e-12, rtol=1e-12)
    gradients = []
    for output in (reference, measured):
        loss = (output.conj() * weight).real.sum() + output.abs().pow(4).sum()
        gradients.append(
            torch.autograd.grad(loss, (state, parameters), create_graph=True)
        )
    for measured_grad, reference_grad in zip(gradients[1], gradients[0], strict=True):
        torch.testing.assert_close(
            measured_grad, reference_grad, atol=1e-10, rtol=1e-11
        )
    second = [
        torch.autograd.grad(
            sum(value.abs().square().sum() for value in gradient), (state, parameters)
        )
        for gradient in gradients
    ]
    for measured_grad, reference_grad in zip(second[1], second[0], strict=True):
        torch.testing.assert_close(measured_grad, reference_grad, atol=1e-8, rtol=1e-10)


@pytest.mark.parametrize("wire", [0, 1, 2])
def test_ry_finite_differences(wire: int) -> None:
    torch.manual_seed(927)
    state = torch.randn(1, 8, dtype=torch.complex128, requires_grad=True)
    parameters = torch.randn(1, 1, dtype=torch.float64, requires_grad=True)

    def apply(state: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return _RYRotation.apply(
            state, parameters, _matrix(parameters).detach(), wire, 3
        )

    assert torch.autograd.gradcheck(apply, (state, parameters))
    assert torch.autograd.gradgradcheck(apply, (state, parameters))
