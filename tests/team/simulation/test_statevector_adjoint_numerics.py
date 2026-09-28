"""Ownership tests for Runtime-independent statevector adjoint numerics."""

import math

import pytest
import torch

from flagquantum.core.ir import Instruction
from flagquantum.simulation.statevector.adjoint import (
    analytic_rotation_derivative,
    real_conjugate_inner_sum,
    z_expectation_adjoint_chunk,
    z_expectation_chunk,
    z_hamiltonian_chunk,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_real_conjugate_inner_sum_matches_complex_definition(dtype):
    left = torch.tensor(
        [[0.25 + 0.5j, -0.75 + 0.125j], [0.33 - 0.2j, -0.4 - 0.6j]],
        dtype=dtype,
    )
    right = torch.tensor(
        [[-0.1 + 0.7j, 0.2 - 0.3j], [0.8 + 0.05j, -0.9 + 0.4j]],
        dtype=dtype,
    )

    actual = real_conjugate_inner_sum(left, right)

    torch.testing.assert_close(actual, torch.real(torch.sum(torch.conj(left) * right)))
    assert actual.dtype == left.real.dtype


def test_z_expectation_chunk_and_adjoint_share_wire_semantics():
    amplitudes = torch.tensor([[0.5 + 0.5j, 0.5 - 0.5j]], dtype=torch.complex64)
    indices = torch.tensor([0, 2])

    expectation = z_expectation_chunk(amplitudes, indices, n_wires=2, wire=0)
    adjoint = z_expectation_adjoint_chunk(amplitudes, indices, n_wires=2, wire=0)

    torch.testing.assert_close(expectation, torch.tensor(0.0))
    torch.testing.assert_close(
        adjoint,
        torch.tensor([[1.0 + 1.0j, -1.0 + 1.0j]], dtype=torch.complex64),
    )


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_z_hamiltonian_chunk_matches_dense_diagonal(dtype):
    amplitudes = torch.tensor(
        [[0.1 + 0.2j, -0.3 + 0.1j, 0.2 - 0.4j, 0.5 + 0.6j]],
        dtype=dtype,
    )
    amplitudes = amplitudes / torch.linalg.vector_norm(amplitudes)
    indices = torch.tensor([0, 2, 5, 7])
    terms = ((0.7, (0, 1)), (0.2, (0,)), (-0.3, (1, 2)))

    value, adjoint = z_hamiltonian_chunk(amplitudes, indices, n_wires=3, terms=terms)

    weights = torch.tensor(
        [
            sum(
                coefficient
                * math.prod(
                    1 - 2 * ((int(index) >> (3 - wire - 1)) & 1) for wire in wires
                )
                for coefficient, wires in terms
            )
            for index in indices
        ],
        dtype=amplitudes.real.dtype,
    )
    expected = (amplitudes.abs().square() * weights).sum()
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(adjoint, 2 * amplitudes * weights)


def _rotation_matrix(
    gate_name: str, theta: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    half = theta / 2
    cosine = torch.cos(half).to(dtype)
    sine = torch.sin(half).to(dtype)
    if gate_name == "rx":
        return torch.stack((cosine, -1j * sine, -1j * sine, cosine)).reshape(2, 2)
    if gate_name == "ry":
        return torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2)
    return torch.diag(
        torch.stack(
            (
                torch.exp(-0.5j * theta).to(dtype),
                torch.exp(0.5j * theta).to(dtype),
            )
        )
    )


@pytest.mark.parametrize("gate_name", ("rx", "ry", "rz"))
@pytest.mark.parametrize("real_dtype", (torch.float32, torch.float64))
def test_standard_rotation_analytic_derivative_matches_autograd(gate_name, real_dtype):
    theta = torch.tensor(0.37, dtype=real_dtype, requires_grad=True)
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    instruction = Instruction(gate_name, (0,), {"theta": theta})

    matrix = _rotation_matrix(gate_name, theta, dtype=complex_dtype)
    _, reference = torch.autograd.functional.jvp(
        lambda value: _rotation_matrix(gate_name, value, dtype=complex_dtype),
        (theta,),
        (torch.ones_like(theta),),
    )
    actual = analytic_rotation_derivative(instruction, matrix)

    assert actual is not None
    assert torch.allclose(actual, reference, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("real_dtype", (torch.float32, torch.float64))
def test_rzz_analytic_derivative_matches_autograd(real_dtype):
    theta = torch.tensor(0.37, dtype=real_dtype, requires_grad=True)
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    instruction = Instruction("rzz", (0, 1), {"theta": theta})

    def matrix(value: torch.Tensor) -> torch.Tensor:
        phases = torch.stack(
            (
                torch.exp(-0.5j * value),
                torch.exp(0.5j * value),
                torch.exp(0.5j * value),
                torch.exp(-0.5j * value),
            )
        ).to(complex_dtype)
        return torch.diag(phases)

    gate_matrix = matrix(theta)
    _, reference = torch.autograd.functional.jvp(
        matrix,
        (theta,),
        (torch.ones_like(theta),),
    )
    actual = analytic_rotation_derivative(instruction, gate_matrix)

    assert actual is not None
    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
