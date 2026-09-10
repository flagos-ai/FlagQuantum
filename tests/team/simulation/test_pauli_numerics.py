from __future__ import annotations

import pytest
import torch

from flagquantum.simulation.pauli import (
    infer_n_wires_from_dense_state,
    pauli_product_density_expectation,
    pauli_product_operator,
    pauli_product_statevector_expectation,
)


def test_pauli_product_matches_for_statevector_density_and_dense_operator() -> None:
    state = torch.tensor(
        [[2**-0.5, 0.0, 0.0, 2**-0.5]],
        dtype=torch.complex128,
    )
    density = state.unsqueeze(-1) * state.conj().unsqueeze(-2)
    operators = ((0, "x"), (1, "x"))

    state_value = pauli_product_statevector_expectation(state, operators, 2)
    density_value = pauli_product_density_expectation(density, operators, 2)
    operator = pauli_product_operator(
        operators,
        2,
        dtype=state.dtype,
        device=state.device,
    )
    direct_value = torch.real(
        torch.einsum("bi,ij,bj->b", state.conj(), operator, state)
    )

    torch.testing.assert_close(state_value, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(density_value, state_value)
    torch.testing.assert_close(direct_value, state_value)


def test_dense_state_dimension_must_be_a_power_of_two() -> None:
    assert infer_n_wires_from_dense_state(torch.zeros(2, 8)) == 3
    with pytest.raises(ValueError, match="power of two"):
        infer_n_wires_from_dense_state(torch.zeros(2, 6))


@pytest.mark.parametrize("name", ("rx", "ry", "rz"))
def test_pauli_products_reject_parameterized_gates(name: str) -> None:
    state = torch.tensor([1, 0], dtype=torch.complex128)
    operators = ((0, name),)
    with pytest.raises(ValueError, match="fixed gate matrices"):
        pauli_product_operator(operators, 1, dtype=state.dtype, device=state.device)
    with pytest.raises(ValueError, match="fixed gate matrices"):
        pauli_product_statevector_expectation(state, operators, 1)


@pytest.mark.parametrize("density_mode", (False, True))
def test_pauli_z_expectation_has_analytic_rotation_gradient(density_mode: bool) -> None:
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    state = torch.stack((torch.cos(theta / 2), torch.sin(theta / 2))).to(
        torch.complex128
    )
    operators = ((0, "z"),)
    if density_mode:
        density = state[:, None] * state.conj()[None, :]
        value = pauli_product_density_expectation(density, operators, 1)
    else:
        value = pauli_product_statevector_expectation(state, operators, 1)

    gradient = torch.autograd.grad(value.sum(), theta)[0]
    torch.testing.assert_close(value.squeeze(), theta.cos())
    torch.testing.assert_close(gradient, -theta.sin())
