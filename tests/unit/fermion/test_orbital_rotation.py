"""Differentiable orbital integral transforms against independent references."""

import pytest
import torch

from flagquantum.algorithms.chemistry_domain.orbital_rotation import (
    closed_shell_determinant_energy,
    rotate_spatial_integrals,
)
from flagquantum.algorithms.chemistry_domain.problem import (
    MolecularProblem,
    fermion_hamiltonian,
)

from .oracles import fock_matrix

pytestmark = pytest.mark.unit


def integrals(n: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(811 + n)
    raw = torch.randn(n, n, generator=generator, dtype=torch.float64)
    one = (raw + raw.T) / 2
    factors = torch.randn(2, n, n, generator=generator, dtype=torch.float64)
    factors = (factors + factors.transpose(1, 2)) / 2
    eri = torch.einsum("lab,lcd->abcd", factors, factors)
    return one, eri


def orbital_generators(n: int) -> torch.Tensor:
    values = []
    for left in range(n):
        for right in range(left + 1, n):
            matrix = torch.zeros(n, n, dtype=torch.float64)
            matrix[left, right] = -1
            matrix[right, left] = 1
            values.append(matrix)
    return torch.stack(values) if values else torch.empty(0, n, n)


def independent_reference_energy(
    one: torch.Tensor, two: torch.Tensor, angle: float, *, constant: float
) -> float:
    generator = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    unitary = torch.linalg.matrix_exp(angle * generator)
    h = unitary.T @ one @ unitary
    eri = torch.einsum(
        "ap,bq,cr,ds,abcd->pqrs", unitary, unitary, unitary, unitary, two
    )
    problem = MolecularProblem(h, eri, 2, constant)
    matrix = fock_matrix(fermion_hamiltonian(problem), 4)
    return float(matrix[12, 12].real)


def test_zero_parameters_reproduce_integrals_and_keep_gradient() -> None:
    one, two = integrals()
    parameters = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    rotated_one, rotated_two = rotate_spatial_integrals(
        one, two, orbital_generators(2)[:1], parameters
    )
    torch.testing.assert_close(rotated_one, one)
    torch.testing.assert_close(rotated_two, two)
    gradient = torch.autograd.grad(rotated_one[0, 1], parameters)[0]
    assert torch.isfinite(gradient).all()


def test_integral_transform_matches_independent_index_contraction() -> None:
    one, two = integrals(3)
    generators = orbital_generators(3)
    parameters = torch.tensor([0.2, -0.31, 0.17], dtype=torch.float64)
    actual_one, actual_two = rotate_spatial_integrals(one, two, generators, parameters)
    unitary = torch.linalg.matrix_exp(torch.einsum("k,kij->ij", parameters, generators))
    expected_one = torch.zeros_like(one)
    expected_two = torch.zeros_like(two)
    for p in range(3):
        for q in range(3):
            for a in range(3):
                for b in range(3):
                    expected_one[p, q] += unitary[a, p] * one[a, b] * unitary[b, q]
            for r in range(3):
                for s in range(3):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                for d in range(3):
                                    expected_two[p, q, r, s] += (
                                        unitary[a, p]
                                        * unitary[b, q]
                                        * unitary[c, r]
                                        * unitary[d, s]
                                        * two[a, b, c, d]
                                    )
    torch.testing.assert_close(actual_one, expected_one)
    torch.testing.assert_close(actual_two, expected_two)
    assert torch.allclose(actual_one, actual_one.T, atol=1e-12, rtol=1e-12)
    for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert torch.allclose(
            actual_two, actual_two.permute(permutation), atol=1e-12, rtol=1e-12
        )


def test_rotated_second_quantized_hamiltonian_preserves_spectrum() -> None:
    one, two = integrals()
    parameters = torch.tensor([0.27], dtype=torch.float64)
    rotated_one, rotated_two = rotate_spatial_integrals(
        one, two, orbital_generators(2)[:1], parameters
    )
    original = MolecularProblem(one, two, 2, 0.19)
    rotated = MolecularProblem(rotated_one, rotated_two, 2, 0.19)
    original_spectrum = torch.linalg.eigvalsh(
        fock_matrix(fermion_hamiltonian(original), 4)
    )
    rotated_spectrum = torch.linalg.eigvalsh(
        fock_matrix(fermion_hamiltonian(rotated), 4)
    )
    torch.testing.assert_close(
        rotated_spectrum, original_spectrum, atol=1e-10, rtol=1e-10
    )


def test_closed_shell_energy_gradient_matches_fock_reference_and_difference() -> None:
    one, two = integrals()
    parameter = torch.tensor([0.23], dtype=torch.float64, requires_grad=True)
    rotated_one, rotated_two = rotate_spatial_integrals(
        one, two, orbital_generators(2)[:1], parameter
    )
    energy = closed_shell_determinant_energy(
        rotated_one, rotated_two, (0,), constant=0.19
    )
    assert float(energy.detach()) == pytest.approx(
        independent_reference_energy(one, two, 0.23, constant=0.19), abs=1e-10
    )
    derivative = torch.autograd.grad(energy, parameter)[0]
    finite_difference = (
        independent_reference_energy(one, two, 0.230001, constant=0.19)
        - independent_reference_energy(one, two, 0.229999, constant=0.19)
    ) / 0.000002
    torch.testing.assert_close(
        derivative,
        torch.tensor([finite_difference], dtype=torch.float64),
        atol=2e-8,
        rtol=2e-7,
    )
    assert float(derivative.abs()) > 1e-3


def test_gradient_matches_torch_gradcheck() -> None:
    one, two = integrals()
    generators = orbital_generators(2)[:1]

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        rotated_one, rotated_two = rotate_spatial_integrals(
            one, two, generators, parameters
        )
        return closed_shell_determinant_energy(rotated_one, rotated_two, (0,))

    assert torch.autograd.gradcheck(
        objective, (torch.tensor([0.13], dtype=torch.float64, requires_grad=True),)
    )


@pytest.mark.parametrize(
    "generators,parameters,match",
    [
        (torch.zeros(2, 2), torch.zeros(1), "3 dimensions"),
        (torch.eye(2).unsqueeze(0), torch.zeros(1), "antisymmetric"),
        (torch.zeros(1, 3, 3), torch.zeros(1), "shape"),
        (torch.zeros(1, 2, 2, requires_grad=True), torch.zeros(1), "autograd"),
        (torch.zeros(1, 2, 2), torch.tensor([float("inf")]), "finite real"),
        (torch.zeros(1, 2, 2), torch.tensor([1], dtype=torch.int64), "finite real"),
        (
            torch.zeros(1, 2, 2),
            torch.zeros(2),
            "finite real",
        ),
    ],
)
def test_rotation_inputs_fail_closed(
    generators: torch.Tensor, parameters: torch.Tensor, match: str
) -> None:
    one, two = integrals()
    with pytest.raises(ValueError, match=match):
        rotate_spatial_integrals(one, two, generators, parameters)


def test_energy_rejects_invalid_occupied_orbitals() -> None:
    one, two = integrals()
    for occupied in ((), (0, 0), (-1,), (2,)):
        with pytest.raises(ValueError, match="occupied orbitals"):
            closed_shell_determinant_energy(one, two, occupied)


def test_reordered_multiple_occupied_energy_matches_fock_expectation() -> None:
    one, two = integrals(3)
    problem = MolecularProblem(one, two, 4, 0.17)
    matrix = fock_matrix(fermion_hamiltonian(problem), 6)
    # Orbitals 0 and 2 doubly occupied: MSB occupation bits 110011.
    expected = matrix[51, 51].real
    actual = closed_shell_determinant_energy(one, two, (2, 0), constant=0.17)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("field", ["one", "two", "generators"])
@pytest.mark.parametrize("failure", ["complex", "nonfinite", "autograd"])
def test_static_rotation_data_rejects_unsupported_inputs(
    field: str, failure: str
) -> None:
    one, two = integrals()
    values = {"one": one, "two": two, "generators": orbital_generators(2)}
    value = values[field]
    if failure == "complex":
        values[field] = value.to(torch.complex128) + 0.1j
    elif failure == "nonfinite":
        value.reshape(-1)[0] = float("nan")
    else:
        value.requires_grad_(True)
    with pytest.raises(ValueError):
        rotate_spatial_integrals(
            values["one"], values["two"], values["generators"], torch.zeros(1)
        )
