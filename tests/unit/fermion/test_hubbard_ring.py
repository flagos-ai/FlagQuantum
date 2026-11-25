"""Periodic Hubbard physics from occupation-bit action and free dispersion."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    fermi_hubbard_chain,
    fermi_hubbard_ring,
)
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from tests.unit.fermion.oracles import fock_matrix

pytestmark = pytest.mark.unit


def _physical_ring(sites: int, t: float, u: float, mu: float) -> torch.Tensor:
    width = 2 * sites
    matrix = torch.zeros((2**width, 2**width), dtype=torch.complex128)
    for column in range(2**width):
        bits = [(column >> (width - 1 - mode)) & 1 for mode in range(width)]
        matrix[column, column] = u * sum(
            bits[2 * i] * bits[2 * i + 1] for i in range(sites)
        ) - mu * sum(bits)
        for source, target in itertools.permutations(range(width), 2):
            if source % 2 != target % 2 or not bits[source] or bits[target]:
                continue
            distance = abs(source // 2 - target // 2)
            if distance not in (1, sites - 1):
                continue
            lo, hi = sorted((source, target))
            sign = (-1) ** sum(bits[lo + 1 : hi])
            row = column ^ (1 << (width - 1 - source)) ^ (1 << (width - 1 - target))
            matrix[row, column] = -t * sign
    return matrix


@pytest.mark.parametrize("sites", [3, 4])
@pytest.mark.parametrize("t,u,mu", [(1.2, 3.1, -0.4), (-0.7, -2.3, 0.6), (0, 2, 0.3)])
def test_complete_action_and_spin_conservation(
    sites: int, t: float, u: float, mu: float
) -> None:
    operator = fermi_hubbard_ring(sites, t=t, u=u, mu=mu)
    actual = fock_matrix(operator, 2 * sites)
    torch.testing.assert_close(actual, _physical_ring(sites, t, u, mu))
    torch.testing.assert_close(actual, actual.mH)
    for spin in (0, 1):
        counts = torch.tensor(
            [
                sum(
                    (i >> (2 * sites - 1 - mode)) & 1
                    for mode in range(spin, 2 * sites, 2)
                )
                for i in range(2 ** (2 * sites))
            ]
        )
        assert torch.count_nonzero(actual * (counts[:, None] - counts[None, :])) == 0
    assert actual[0, 0].item() == 0
    assert actual[-1, -1].real.item() == pytest.approx(sites * u - 2 * sites * mu)


@pytest.mark.parametrize("spectator,expected", [(None, -1.0), (1, 1.0), (2, 1.0)])
def test_wrap_hopping_sign_depends_on_occupied_modes_between_endpoints(
    spectator: int | None, expected: float
) -> None:
    # Transfer spin up from mode 4 (last site) to mode 0 (first site).
    occupied = [4] if spectator is None else [4, spectator]
    column = sum(1 << (5 - q) for q in occupied)
    row = column ^ (1 << 1) ^ (1 << 5)
    actual = fock_matrix(fermi_hubbard_ring(3), 6)
    assert actual[row, column].real.item() == expected
    open_chain = fock_matrix(fermi_hubbard_chain(3), 6)
    assert open_chain[row, column].item() == 0


@pytest.mark.parametrize("sites,t,mu", [(3, 1.1, -0.3), (4, -0.7, 0.8)])
def test_free_dispersion_in_every_particle_sector(
    sites: int, t: float, mu: float
) -> None:
    levels = [
        -2 * t * math.cos(2 * math.pi * k / sites) - mu
        for k in range(sites)
        for _ in range(2)
    ]
    width = 2 * sites
    matrix = fock_matrix(fermi_hubbard_ring(sites, t=t, mu=mu), width)
    for population in range(width + 1):
        sector = [i for i in range(2**width) if i.bit_count() == population]
        actual = torch.linalg.eigvalsh(matrix[sector][:, sector])
        expected = torch.tensor(
            sorted(sum(c) for c in itertools.combinations(levels, population)),
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_three_encodings_preserve_boundary_terms_and_explicit_zero_width() -> None:
    physical = _physical_ring(3, -0.7, 2.1, 0.4)
    for encoding in ENCODINGS:
        encoded = encode_fermion(
            fermi_hubbard_ring(3, t=-0.7, u=2.1, mu=0.4), n_modes=6, encoding=encoding
        )
        indices = []
        for index in range(64):
            bits = encoded.reference_bits(
                tuple((index >> (5 - q)) & 1 for q in range(6))
            )
            indices.append(sum(bit << (5 - q) for q, bit in enumerate(bits)))
        matrix = pauli_matrix(encoded.operator, n_qubits=6)
        torch.testing.assert_close(matrix[indices][:, indices], physical)
        zero = encode_fermion(fermi_hubbard_ring(3, t=0), n_modes=6, encoding=encoding)
        assert zero.n_modes == 6
        assert zero.operator.terms == ()


@pytest.mark.parametrize("sites", [0, -1, 1, 2])
def test_ambiguous_small_cycles_are_refused(sites: int) -> None:
    with pytest.raises(ValueError, match="n_sites"):
        fermi_hubbard_ring(sites)


@pytest.mark.parametrize("sites", [True, 3.0, "3", 3 + 0j])
def test_wrong_site_type(sites: int) -> None:
    with pytest.raises(TypeError):
        fermi_hubbard_ring(sites)


@pytest.mark.parametrize("name", ["t", "u", "mu"])
@pytest.mark.parametrize(
    "value",
    [True, 1 + 0j, "1", None, torch.tensor(1.0), torch.tensor(1.0, requires_grad=True)],
)
def test_static_coefficients_only(name: str, value: float) -> None:
    with pytest.raises(TypeError):
        fermi_hubbard_ring(3, **{name: value})


@pytest.mark.parametrize("name", ["t", "u", "mu"])
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_coefficients(name: str, value: float) -> None:
    with pytest.raises(ValueError):
        fermi_hubbard_ring(3, **{name: value})
