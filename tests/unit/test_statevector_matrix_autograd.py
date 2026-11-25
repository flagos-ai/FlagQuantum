"""Verify complex gate contractions and their higher-order gradients."""

import pytest
import torch

from flagquantum.simulation.statevector.matrix_autograd import _GateMatrixBMM

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("batch,length", [(1, 17), (2, 31), (1, 65536)])
@pytest.mark.parametrize("strided", [False, True])
def test_complex_gate_gradients(batch: int, length: int, strided: bool) -> None:
    torch.manual_seed(131)
    matrix = torch.randn(batch, 2, 2, dtype=torch.complex128, requires_grad=True)
    raw = torch.randn(batch, length, 2, dtype=torch.complex128)
    amplitudes = raw.transpose(1, 2)
    if not strided:
        amplitudes = amplitudes.contiguous()
    amplitudes.requires_grad_()
    weight = torch.randn(batch, 2, length, dtype=torch.complex128)
    reference = torch.bmm(matrix, amplitudes)
    measured = _GateMatrixBMM.apply(matrix, amplitudes)
    torch.testing.assert_close(measured, reference, atol=1e-12, rtol=1e-12)
    expected_gradients = torch.autograd.grad(
        (reference.conj() * weight).real.sum(),
        (matrix, amplitudes),
        create_graph=True,
    )
    actual_gradients = torch.autograd.grad(
        (measured.conj() * weight).real.sum(),
        (matrix, amplitudes),
        create_graph=True,
    )
    for expected, actual in zip(expected_gradients, actual_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-12)
    expected_second = torch.autograd.grad(
        sum(gradient.abs().square().sum() for gradient in expected_gradients),
        (matrix, amplitudes),
    )
    actual_second = torch.autograd.grad(
        sum(gradient.abs().square().sum() for gradient in actual_gradients),
        (matrix, amplitudes),
    )
    for expected, actual in zip(expected_second, actual_second, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-10)


def test_complex_gate_finite_difference_gradients() -> None:
    torch.manual_seed(331)
    matrix = torch.randn(1, 2, 2, dtype=torch.complex128, requires_grad=True)
    amplitudes = torch.randn(1, 2, 3, dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(_GateMatrixBMM.apply, (matrix, amplitudes))
    assert torch.autograd.gradgradcheck(_GateMatrixBMM.apply, (matrix, amplitudes))
