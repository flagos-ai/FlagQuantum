"""Protect exact Pauli expectations and compact permutation derivatives."""

import pytest
import torch

from flagquantum.simulation.pauli import pauli_product_operator
from flagquantum.simulation.statevector.expectation_autograd import _PauliExpectation
from flagquantum.simulation.statevector.operations import _apply_cx_permutation
from flagquantum.simulation.statevector.permutation_autograd import (
    apply_compact_cx_sequence,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    "operators", [(), ((0, "z"),), ((1, "Y"),), ((0, "x"), (2, "y"), (3, "z"))]
)
def test_hermitian_expectation_derivatives(batch, operators):
    torch.manual_seed(815)
    state = torch.randn(batch, 16, dtype=torch.complex128, requires_grad=True)
    matrix = pauli_product_operator(operators, 4, dtype=state.dtype, device="cpu")
    reference = (state.conj() * (state @ matrix.T)).sum(-1).real
    measured = _PauliExpectation.apply(state, operators, 4)
    torch.testing.assert_close(measured, reference, atol=1e-12, rtol=1e-12)
    gradients = [
        torch.autograd.grad(x.square().sum(), state, create_graph=True)[0]
        for x in (reference, measured)
    ]
    torch.testing.assert_close(gradients[1], gradients[0], atol=1e-10, rtol=1e-11)
    second = [torch.autograd.grad(x.abs().square().sum(), state)[0] for x in gradients]
    torch.testing.assert_close(second[1], second[0], atol=1e-8, rtol=1e-10)


@pytest.mark.parametrize(
    "controls,targets,n_wires",
    [((0, 1), (1, 2), 4), ((1, 3, 1), (3, 2, 3), 4), ((8, 9), (9, 10), 11)],
)
def test_compact_permutation_inverse(controls, targets, n_wires):
    torch.manual_seed(991)
    state = torch.randn(2, 1 << n_wires, dtype=torch.complex128, requires_grad=True)
    reference = state
    for control, target in zip(controls, targets, strict=True):
        reference = _apply_cx_permutation(reference, (control, target), n_wires)
    measured = apply_compact_cx_sequence(state, controls, targets, n_wires)
    torch.testing.assert_close(measured, reference, atol=0, rtol=0)
    weight = torch.randn_like(state)
    ref_grad = torch.autograd.grad((reference.conj() * weight).real.sum(), state)[0]
    got_grad = torch.autograd.grad((measured.conj() * weight).real.sum(), state)[0]
    torch.testing.assert_close(got_grad, ref_grad, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["permutation", "expectation"])
def test_hermitian_finite_differences(kind):
    state = torch.randn(1, 8, dtype=torch.complex128, requires_grad=True)

    def apply(state):
        if kind == "permutation":
            return apply_compact_cx_sequence(state, (0, 1), (1, 2), 3)
        return _PauliExpectation.apply(state, ((0, "y"), (2, "z")), 3)

    assert torch.autograd.gradcheck(apply, (state,))
    assert torch.autograd.gradgradcheck(apply, (state,))
