"""Spinless t-V models checked by occupation action and independent spectra."""

import itertools
import math

import pytest
import torch

from flagquantum.algorithms.fermion.encoding import ENCODINGS, encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    spinless_fermion_chain,
    spinless_fermion_ring,
)
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from tests.unit.fermion.oracles import fock_matrix

pytestmark = pytest.mark.unit


def _physical(
    sites: int, periodic: bool, t: float, v: float, mu: float
) -> torch.Tensor:
    matrix = torch.zeros((2**sites, 2**sites), dtype=torch.complex128)
    for column in range(2**sites):
        occupied = [(column >> (sites - 1 - q)) & 1 for q in range(sites)]
        matrix[column, column] = -mu * sum(occupied)
        for left in range(sites):
            for right in range(left + 1, sites):
                if right - left != 1 and not (
                    periodic and left == 0 and right == sites - 1
                ):
                    continue
                matrix[column, column] += v * occupied[left] * occupied[right]
                if occupied[left] != occupied[right]:
                    row = (
                        column ^ (1 << (sites - 1 - left)) ^ (1 << (sites - 1 - right))
                    )
                    matrix[row, column] = -t * (-1) ** sum(occupied[left + 1 : right])
    return matrix


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("sites", [3, 4, 5])
@pytest.mark.parametrize("t,v,mu", [(1.2, 2.3, -0.4), (-0.7, -1.3, 0.2), (0, 1.7, 0.6)])
def test_full_action_interaction_and_number_conservation(
    sites: int, periodic: bool, t: float, v: float, mu: float
) -> None:
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    actual = fock_matrix(builder(sites, t=t, v=v, mu=mu), sites)
    torch.testing.assert_close(
        actual, _physical(sites, periodic, t, v, mu), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(actual, actual.mH)
    population = torch.tensor([i.bit_count() for i in range(2**sites)])
    assert (
        torch.count_nonzero(actual * (population[:, None] - population[None, :])) == 0
    )
    bonds = sites if periodic else sites - 1
    assert actual[-1, -1].real.item() == pytest.approx(bonds * v - sites * mu)
    assert actual[0, 0].item() == 0


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("sites", [3, 4, 5])
def test_free_spectrum_in_every_particle_sector(periodic: bool, sites: int) -> None:
    t, mu = -0.7, 0.3
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    levels = [
        -2
        * t
        * math.cos(
            2 * math.pi * k / sites if periodic else math.pi * (k + 1) / (sites + 1)
        )
        - mu
        for k in range(sites)
    ]
    matrix = fock_matrix(builder(sites, t=t, mu=mu), sites)
    for n in range(sites + 1):
        indices = [i for i in range(2**sites) if i.bit_count() == n]
        expected = torch.tensor(
            sorted(sum(c) for c in itertools.combinations(levels, n)),
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            torch.linalg.eigvalsh(matrix[indices][:, indices]),
            expected,
            atol=1e-12,
            rtol=0,
        )


@pytest.mark.parametrize("spectators", [(), (1,), (1, 2)])
def test_ring_closing_sign_and_density(spectators: tuple[int, ...]) -> None:
    sites = 4
    column = sum(1 << (3 - q) for q in (*spectators, 3))
    row = column ^ 1 ^ 8
    ring = fock_matrix(spinless_fermion_ring(sites), sites)
    chain = fock_matrix(spinless_fermion_chain(sites), sites)
    assert ring[row, column].item() == -((-1) ** len(spectators))
    assert chain[row, column].item() == 0
    density = fock_matrix(
        spinless_fermion_ring(4, t=0, v=2) - spinless_fermion_chain(4, t=0, v=2), 4
    )
    assert density[9, 9].item() == 2
    assert density[1, 1].item() == 0


@pytest.mark.parametrize("periodic", [False, True])
def test_all_encodings_and_zero_width(periodic: bool) -> None:
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    for encoding in ENCODINGS:
        encoded = encode_fermion(
            builder(4, t=0.7, v=2.1, mu=-0.2), n_modes=4, encoding=encoding
        )
        indices = [
            sum(
                bit << (3 - q)
                for q, bit in enumerate(
                    encoded.reference_bits(tuple((i >> (3 - j)) & 1 for j in range(4)))
                )
            )
            for i in range(16)
        ]
        matrix = pauli_matrix(encoded.operator, n_qubits=4)
        torch.testing.assert_close(
            matrix[indices][:, indices], _physical(4, periodic, 0.7, 2.1, -0.2)
        )
        zero = encode_fermion(builder(4, t=0), n_modes=4, encoding=encoding)
        assert zero.n_modes == 4 and zero.operator.terms == ()


def test_one_site_chain_and_two_site_single_bond() -> None:
    torch.testing.assert_close(
        fock_matrix(spinless_fermion_chain(1, v=8, mu=0.3), 1),
        torch.diag(torch.tensor([0, -0.3], dtype=torch.complex128)),
    )
    matrix = fock_matrix(spinless_fermion_chain(2, t=1.2, v=3, mu=0.2), 2)
    assert matrix[1, 2].item() == -1.2
    assert matrix[3, 3].real.item() == pytest.approx(2.6)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("sites", [True, 3.0, "3", 3 + 0j])
def test_wrong_site_type(periodic: bool, sites: int) -> None:
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    with pytest.raises(TypeError):
        builder(sites)


@pytest.mark.parametrize(
    "periodic,sites", [(False, 0), (False, -1), (True, 0), (True, 1), (True, 2)]
)
def test_invalid_site_count(periodic: bool, sites: int) -> None:
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    with pytest.raises(ValueError, match="n_sites"):
        builder(sites)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("name", ["t", "v", "mu"])
@pytest.mark.parametrize("value", [True, 1j, torch.tensor(1.0), "1"])
def test_wrong_coefficient_type(periodic: bool, name: str, value: float) -> None:
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    with pytest.raises(TypeError, match=name):
        builder(3, **{name: value})


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("name", ["t", "v", "mu"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_coefficients(periodic: bool, name: str, value: float) -> None:
    builder = spinless_fermion_ring if periodic else spinless_fermion_chain
    with pytest.raises(ValueError):
        builder(3, **{name: value})
