"""Exact Krylov closure preserves values and transverse first variations."""

from __future__ import annotations

import pytest
import torch

from flagquantum.simulation.matrix_free_exponential import exponential_action
from flagquantum.simulation.matrix_free_linear_solve import gmres

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["solve", "exponential"])
@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_closed_subspace_derivatives_match_dense_torch(method, dimension, precision):
    dtype = getattr(torch, precision)
    tolerance = 1e-5 if precision == "complex64" else 1e-12
    base = torch.tensor(
        [[1.0 + 0.25j, 1.0, 0.0], [0.0, 2.0 - 0.5j, 0.0], [0.0, 0.0, 3.0 + 0.2j]],
        dtype=dtype,
    )
    perturbation = torch.tensor(
        [[0.2j, 0.1, -0.2], [0.3, -0.1j, 0.4], [1.0j, 0.7, 0.1]], dtype=dtype
    )
    parameter = torch.tensor(0.0j, dtype=dtype, requires_grad=True)
    vector = torch.zeros(3, dtype=dtype)
    vector[dimension - 1] = 1
    vector.requires_grad_()
    matrix = base + parameter * perturbation

    def run():
        if method == "solve":
            return gmres(
                lambda value: matrix @ value,
                vector,
                tolerance=tolerance,
                max_iterations=64,
                restart=8,
            )[0]
        return exponential_action(
            lambda value: matrix @ value,
            0.3,
            vector,
            tolerance=tolerance,
            max_iterations=64,
            restart=8,
        )[0]

    result = run()
    with torch.no_grad():
        unchanged = run()
    assert torch.equal(result.detach().view(torch.uint8), unchanged.view(torch.uint8))
    cost = result[1].real + 0.3 * result[2].imag
    actual = torch.autograd.grad(cost, (parameter, vector))
    reference_matrix = base + parameter * perturbation
    reference = (
        torch.linalg.solve(reference_matrix, vector)
        if method == "solve"
        else torch.matrix_exp(0.3 * reference_matrix) @ vector
    )
    expected = torch.autograd.grad(
        reference[1].real + 0.3 * reference[2].imag, (parameter, vector)
    )
    for measured, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            measured, target, atol=2e-6 if precision == "complex64" else 1e-11, rtol=0
        )


@pytest.mark.parametrize("method", ["solve", "exponential"])
def test_a_zero_argument_retains_its_linear_input_derivative(method):
    matrix = torch.tensor([[1.0, 0.2j], [0.0, 2.0]], dtype=torch.complex128)
    argument = torch.zeros(2, dtype=torch.complex128, requires_grad=True)
    kwargs = dict(tolerance=1e-12, max_iterations=64, restart=8)
    if method == "solve":
        result = gmres(lambda value: matrix @ value, argument, **kwargs)[0]
        reference = torch.linalg.solve(matrix, argument)
    else:
        result = exponential_action(
            lambda value: matrix @ value, 0.3, argument, **kwargs
        )[0]
        reference = torch.matrix_exp(0.3 * matrix) @ argument
    (actual,) = torch.autograd.grad(result.real.sum(), argument)
    (expected,) = torch.autograd.grad(reference.real.sum(), argument)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


def test_closed_subspace_correction_does_not_claim_higher_order_derivatives():
    parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    matrix = torch.eye(2, dtype=torch.complex128) + parameter * torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128
    )
    result = gmres(
        lambda value: matrix @ value,
        torch.tensor([1.0, 0.0], dtype=torch.complex128),
        tolerance=1e-12,
        max_iterations=64,
        restart=8,
    )[0]
    with pytest.raises(RuntimeError, match="higher-order"):
        torch.autograd.grad(result[1].real, parameter, create_graph=True)


@pytest.mark.parametrize("method", ["solve", "exponential"])
def test_sensitivity_that_exceeds_the_same_krylov_budget_fails_closed(method):
    parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex128))
    matrix = matrix + parameter * torch.ones_like(matrix)
    vector = torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)
    kwargs = dict(tolerance=1e-12, max_iterations=1, restart=1)
    result = (
        gmres(lambda value: matrix @ value, vector, **kwargs)[0]
        if method == "solve"
        else exponential_action(lambda value: matrix @ value, 0.3, vector, **kwargs)[0]
    )
    with pytest.raises(RuntimeError, match="adjoint.*converge"):
        torch.autograd.grad(result[1:].real.sum(), parameter)


@pytest.mark.parametrize("method", ["solve", "exponential"])
def test_a_complete_basis_needs_no_transverse_correction_and_keeps_its_graph(method):
    parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    matrix = torch.tensor([[1.0, 0.0], [1.0, 2.0]], dtype=torch.complex128)
    matrix = matrix + parameter * torch.diag(
        torch.tensor([1.0, 0.0], dtype=torch.complex128)
    )
    vector = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    kwargs = dict(tolerance=1e-12, max_iterations=64, restart=8)
    if method == "solve":
        result = gmres(lambda value: matrix @ value, vector, **kwargs)[0]
        reference = torch.linalg.solve(matrix, vector)
    else:
        result = exponential_action(
            lambda value: matrix @ value, 0.3, vector, **kwargs
        )[0]
        reference = torch.matrix_exp(0.3 * matrix) @ vector
    (first,) = torch.autograd.grad(result[1].real, parameter, create_graph=True)
    (expected_first,) = torch.autograd.grad(
        reference[1].real, parameter, create_graph=True
    )
    (second,) = torch.autograd.grad(first, parameter, retain_graph=True)
    (expected_second,) = torch.autograd.grad(expected_first, parameter)
    torch.testing.assert_close(first, expected_first, atol=1e-11, rtol=0)
    torch.testing.assert_close(second, expected_second, atol=1e-11, rtol=0)
