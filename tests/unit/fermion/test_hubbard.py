"""Open-chain physics against occupation hopping and analytic sector spectra."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_chain
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from tests.unit.fermion.oracles import fock_matrix

pytestmark = pytest.mark.unit


def _physical_matrix(sites: int, t: float, u: float, mu: float) -> torch.Tensor:
    width = 2 * sites
    matrix = torch.zeros((2**width, 2**width), dtype=torch.complex128)
    for column in range(2**width):
        bits = [(column >> (width - 1 - mode)) & 1 for mode in range(width)]
        matrix[column, column] = u * sum(
            bits[2 * i] * bits[2 * i + 1] for i in range(sites)
        ) - mu * sum(bits)
        # Move either spin both ways; the sign counts occupied modes between them.
        for source, target in itertools.permutations(range(width), 2):
            if abs(source - target) != 2 or not bits[source] or bits[target]:
                continue
            lo, hi = sorted((source, target))
            sign = (-1) ** sum(bits[lo + 1 : hi])
            row = column ^ (1 << (width - 1 - source)) ^ (1 << (width - 1 - target))
            matrix[row, column] = -t * sign
    return matrix


def _sector(width: int, population: int) -> list[int]:
    return [i for i in range(2**width) if i.bit_count() == population]


@pytest.mark.parametrize("sites", [1, 2, 3])
@pytest.mark.parametrize("t,u,mu", [(1.2, 3.1, -0.4), (-0.7, -2.3, 0.6), (0, 0, 0)])
def test_full_matrix_and_conserved_populations(
    sites: int, t: float, u: float, mu: float
) -> None:
    width = 2 * sites
    operator = fermi_hubbard_chain(sites, t=t, u=u, mu=mu)
    actual = fock_matrix(operator, width)
    torch.testing.assert_close(actual, _physical_matrix(sites, t, u, mu))
    torch.testing.assert_close(actual, actual.mH)
    for spin in (0, 1):
        counts = torch.tensor(
            [
                sum((i >> (width - 1 - mode)) & 1 for mode in range(spin, width, 2))
                for i in range(2**width)
            ]
        )
        assert torch.count_nonzero(actual * (counts[:, None] - counts[None, :])) == 0
    assert actual[0, 0] == 0
    assert actual[-1, -1] == pytest.approx(sites * u - width * mu)


@pytest.mark.parametrize("t,u", [(1, 4), (-1.3, 2.7), (0.6, -3), (0, -2), (0, 2)])
def test_two_site_two_particle_ground_energy(t: float, u: float) -> None:
    matrix = fock_matrix(fermi_hubbard_chain(2, t=t, u=u), 4)
    indices = _sector(4, 2)
    energy = torch.linalg.eigvalsh(matrix[indices][:, indices])[0].item()
    assert energy == pytest.approx((u - math.sqrt(u**2 + 16 * t**2)) / 2, abs=1e-12)


@pytest.mark.parametrize("sites,t,mu", [(2, 1.1, -0.3), (3, -0.7, 0.8)])
def test_free_chain_spectrum_in_every_particle_sector(
    sites: int, t: float, mu: float
) -> None:
    levels = [
        -2 * t * math.cos(k * math.pi / (sites + 1)) - mu
        for k in range(1, sites + 1)
        for _ in range(2)
    ]
    width = 2 * sites
    matrix = fock_matrix(fermi_hubbard_chain(sites, t=t, mu=mu), width)
    for population in range(width + 1):
        indices = _sector(width, population)
        actual = torch.linalg.eigvalsh(matrix[indices][:, indices])
        expected = torch.tensor(
            sorted(sum(c) for c in itertools.combinations(levels, population)),
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_encodings_preserve_complete_hamiltonian_in_their_occupation_basis() -> None:
    operator = fermi_hubbard_chain(3, t=-0.7, u=2.1, mu=0.4)
    physical = _physical_matrix(3, -0.7, 2.1, 0.4)
    for encoding in ENCODINGS:
        encoded = encode_fermion(operator, n_modes=6, encoding=encoding)
        permutation = []
        for i in range(64):
            occupation = tuple((i >> (5 - mode)) & 1 for mode in range(6))
            bits = encoded.reference_bits(occupation)
            permutation.append(sum(bit << (5 - q) for q, bit in enumerate(bits)))
        actual = pauli_matrix(encoded.operator, n_qubits=6)
        torch.testing.assert_close(actual[permutation][:, permutation], physical)


def test_one_site_and_zero_operator_keep_explicit_encoding_width() -> None:
    matrix = fock_matrix(fermi_hubbard_chain(1, t=100, u=4, mu=0.5), 2)
    torch.testing.assert_close(
        matrix.diag().real, torch.tensor([0, -0.5, -0.5, 3], dtype=torch.float64)
    )
    zero = fermi_hubbard_chain(3, t=0)
    assert zero.terms == ()
    for encoding in ENCODINGS:
        encoded = encode_fermion(zero, n_modes=6, encoding=encoding)
        assert encoded.n_modes == 6
        assert torch.count_nonzero(pauli_matrix(encoded.operator, n_qubits=6)) == 0


@pytest.mark.parametrize("sites", [True, 2.0, "2", 1 + 0j])
def test_wrong_site_type(sites: int) -> None:
    with pytest.raises(TypeError):
        fermi_hubbard_chain(sites)


@pytest.mark.parametrize("sites", [0, -1])
def test_nonpositive_site_count(sites: int) -> None:
    with pytest.raises(ValueError):
        fermi_hubbard_chain(sites)


@pytest.mark.parametrize("name", ["t", "u", "mu"])
@pytest.mark.parametrize(
    "value",
    [True, 1 + 0j, "1", None, torch.tensor(1.0), torch.tensor(1.0, requires_grad=True)],
)
def test_wrong_coefficient_type(name: str, value: float) -> None:
    with pytest.raises(TypeError):
        fermi_hubbard_chain(2, **{name: value})


@pytest.mark.parametrize("name", ["t", "u", "mu"])
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_coefficient(name: str, value: float) -> None:
    with pytest.raises(ValueError):
        fermi_hubbard_chain(2, **{name: value})
