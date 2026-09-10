"""Direct trace contraction preserves complex density expectations and gradients."""

import pytest
import torch

from flagquantum.algorithms import pauli_term
from flagquantum.simulation.pauli import pauli_product_density_expectation

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("wire", [-1, 2])
def test_density_expectation_rejects_out_of_range_operator(wire: int) -> None:
    density = torch.eye(4, dtype=torch.complex128) / 4
    with pytest.raises(ValueError, match="wire index out of range"):
        pauli_term(1.0, "Z", (wire,)).expectation(density)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_density_pauli_trace_matches_matrix_product_and_gradient(
    batched: bool, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(214)
    states = torch.randn(3, 4, generator=generator, dtype=dtype)
    states = states / torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    density = states.unsqueeze(-1) * states.conj().unsqueeze(-2)
    if not batched:
        density = density[0]
    density.requires_grad_(True)
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    operator = torch.kron(y, x)
    batch = density if batched else density.unsqueeze(0)
    expected = torch.stack([torch.trace(item @ operator).real for item in batch])
    actual = pauli_product_density_expectation(density, ((0, "y"), (1, "x")), 2)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.square().sum(), density)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), density)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
