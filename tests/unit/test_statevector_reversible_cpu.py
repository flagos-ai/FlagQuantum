"""Exact CPU Toffoli execution preserves amplitudes, ownership and gradients."""

from itertools import permutations

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.matrices import TOFFOLI_MATRIX
from flagquantum.simulation.statevector.operations import (
    _apply_fixed_permutation,
    _apply_matrix_layout,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("wires", list(permutations(range(4), 3)))
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cpu_toffoli_matches_dense_on_every_basis_state(
    wires: tuple[int, int, int], dtype: torch.dtype
) -> None:
    state = torch.eye(16, dtype=dtype)
    actual = _apply_fixed_permutation(state, "ccx", wires, 4)
    expected = _apply_matrix_layout(state, TOFFOLI_MATRIX.to(dtype), wires, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, torch.eye(16, dtype=dtype), rtol=0, atol=0)
    assert actual.data_ptr() != state.data_ptr()


@pytest.mark.parametrize("wires", ((0, 2, 4), (4, 2, 0), (3, 0, 2)))
@pytest.mark.parametrize("layout", ("contiguous", "strided", "conjugate"))
def test_cpu_toffoli_matches_dense_complex_gradient_and_inverse(
    wires: tuple[int, int, int], layout: str
) -> None:
    generator = torch.Generator().manual_seed(1253)
    source = torch.randn(3, 64, dtype=torch.complex128, generator=generator)
    state = source[:, :32].clone() if layout == "contiguous" else source[:, ::2]
    if layout == "conjugate":
        state = state.conj()
    state = state.detach().requires_grad_()
    before = state.detach().clone()
    reference = before.clone().requires_grad_()
    actual = _apply_fixed_permutation(state, "ccx", wires, 5)
    expected = _apply_matrix_layout(reference, TOFFOLI_MATRIX, wires, 5)
    weights = torch.randn(3, 32, dtype=torch.complex128, generator=generator)
    actual_gradient = torch.autograd.grad((actual * weights).real.sum(), state)[0]
    expected_gradient = torch.autograd.grad((expected * weights).real.sum(), reference)[
        0
    ]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(state, before, rtol=0, atol=0)
    torch.testing.assert_close(
        _apply_fixed_permutation(actual, "ccx", wires, 5), before, rtol=0, atol=0
    )


def test_cpu_public_toffoli_uses_permutation_without_dense_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = torch.arange(32, dtype=torch.float64).to(torch.complex128)
    circuit = fq.Circuit(5, dtype=torch.complex128, inputs=initial).ccx(4, 1, 2)
    expected = _apply_matrix_layout(initial[None], TOFFOLI_MATRIX, (4, 1, 2), 5)

    def reject_bmm(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("CPU Toffoli must not construct a dense multiplication")

    monkeypatch.setattr(torch, "bmm", reject_bmm)
    torch.testing.assert_close(circuit.state(), expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, torch.arange(32).to(torch.complex128))
