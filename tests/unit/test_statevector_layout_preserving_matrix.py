"""Protect arbitrary complex gate gradients across wire layouts."""

import pytest
import torch

from flagquantum.simulation.statevector.matrix_autograd import _SingleWireMatrix
from flagquantum.simulation.statevector.operations import _apply_matrix_layout

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("batch,shared", [(1, True), (2, True), (2, False)])
@pytest.mark.parametrize("wire", [0, 1, 3, 5])
def test_wire_matrix_derivatives(batch: int, shared: bool, wire: int) -> None:
    torch.manual_seed(727)
    state = torch.randn(batch, 64, dtype=torch.complex128, requires_grad=True)
    shape = (2, 2) if shared else (batch, 2, 2)
    matrix = torch.randn(shape, dtype=torch.complex128, requires_grad=True)
    reference = _apply_matrix_layout(state, matrix, (wire,), 6)
    measured = _SingleWireMatrix.apply(state, matrix, wire, 6)
    torch.testing.assert_close(measured, reference, atol=1e-12, rtol=1e-12)
    expected = torch.autograd.grad(
        reference.abs().square().sum(), (state, matrix), create_graph=True
    )
    actual = torch.autograd.grad(
        measured.abs().square().sum(), (state, matrix), create_graph=True
    )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-12)
    expected_second = torch.autograd.grad(
        sum(x.abs().square().sum() for x in expected), (state, matrix)
    )
    actual_second = torch.autograd.grad(
        sum(x.abs().square().sum() for x in actual), (state, matrix)
    )
    for a, b in zip(actual_second, expected_second, strict=True):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-11)


@pytest.mark.parametrize("wire", [0, 1, 2])
def test_wire_matrix_finite_differences(wire: int) -> None:
    torch.manual_seed(927)
    state = torch.randn(1, 8, dtype=torch.complex128, requires_grad=True)
    matrix = torch.randn(2, 2, dtype=torch.complex128, requires_grad=True)

    def apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _SingleWireMatrix.apply(a, b, wire, 3)

    assert torch.autograd.gradcheck(apply, (state, matrix))
    assert torch.autograd.gradgradcheck(apply, (state, matrix))
