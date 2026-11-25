"""Explicit graph models against independent occupation-bit hopping."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch

from flagquantum.algorithms.fermion.encoding import ENCODINGS, Encoding, encode_fermion
from flagquantum.algorithms.fermion.hubbard import (
    fermi_hubbard_chain,
    fermi_hubbard_graph,
    fermi_hubbard_ring,
)
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from tests.unit.fermion.oracles import fock_matrix

pytestmark = pytest.mark.unit


def occupation_matrix(
    sites: int, edges: Sequence[tuple[int, int]], t: float, u: float, mu: float
) -> torch.Tensor:
    width = 2 * sites
    result = torch.zeros((2**width, 2**width), dtype=torch.complex128)
    for column in range(2**width):
        bits = [(column >> (width - 1 - q)) & 1 for q in range(width)]
        result[column, column] = u * sum(
            bits[2 * site] * bits[2 * site + 1] for site in range(sites)
        ) - mu * sum(bits)
        for left, right in edges:
            for spin in (0, 1):
                a, b = 2 * left + spin, 2 * right + spin
                if bits[a] == bits[b]:
                    continue
                lo, hi = sorted((a, b))
                sign = (-1) ** sum(bits[lo + 1 : hi])
                row = column ^ (1 << (width - 1 - a)) ^ (1 << (width - 1 - b))
                result[row, column] = -t * sign
    return result


def _code_indices(width: int, encoding: Encoding) -> list[int]:
    indices = []
    for index in range(2**width):
        occupation = [(index >> (width - 1 - q)) & 1 for q in range(width)]
        code = []
        for q in range(width):
            if encoding == "jordan_wigner":
                start = q
            elif encoding == "parity":
                start = 0
            else:
                start = q + 1 - ((q + 1) & -(q + 1))
            code.append(sum(occupation[start : q + 1]) % 2)
        indices.append(sum(bit << (width - 1 - q) for q, bit in enumerate(code)))
    return indices


@pytest.mark.parametrize(
    "sites,edges", [(4, ((0, 1), (0, 2), (0, 3))), (4, ((0, 2),)), (1, ())]
)
@pytest.mark.parametrize("t,u,mu", [(1.2, 3.1, -0.4), (-0.7, -2.3, 0.6), (0, 0, 0)])
def test_graph_action_and_separate_spin_conservation(
    sites: int, edges: tuple[tuple[int, int], ...], t: float, u: float, mu: float
) -> None:
    width = 2 * sites
    operator = fermi_hubbard_graph(sites, edges, t=t, u=u, mu=mu)
    reordered = tuple((b, a) for a, b in reversed(edges))
    assert operator == fermi_hubbard_graph(sites, reordered, t=t, u=u, mu=mu)
    actual = fock_matrix(operator, width)
    torch.testing.assert_close(actual, occupation_matrix(sites, edges, t, u, mu))
    torch.testing.assert_close(actual, actual.mH)
    for spin in (0, 1):
        counts = torch.tensor(
            [
                sum((i >> (width - 1 - q)) & 1 for q in range(spin, width, 2))
                for i in range(2**width)
            ]
        )
        assert torch.count_nonzero(actual * (counts[:, None] - counts[None, :])) == 0


@pytest.mark.parametrize("sites", [1, 2, 3, 4])
def test_path_and_cycle_replace_existing_model_producers(sites: int) -> None:
    path = tuple((q, q + 1) for q in range(sites - 1))
    assert fermi_hubbard_graph(sites, path, t=-0.7, u=2, mu=0.3) == fermi_hubbard_chain(
        sites, t=-0.7, u=2, mu=0.3
    )
    if sites >= 3:
        cycle = (*path, (sites - 1, 0))
        assert fermi_hubbard_graph(sites, cycle, u=4) == fermi_hubbard_ring(sites, u=4)
    reverse = tuple((b, a) for a, b in reversed(path))
    assert fermi_hubbard_graph(sites, path) == fermi_hubbard_graph(sites, reverse)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_branching_graph_reaches_each_encoding(encoding: Encoding) -> None:
    edges = ((0, 1), (0, 2), (0, 3))
    encoded = encode_fermion(
        fermi_hubbard_graph(4, edges, t=0.7, u=2, mu=-0.3),
        n_modes=8,
        encoding=encoding,
    )
    indices = _code_indices(8, encoding)
    actual = pauli_matrix(encoded.operator, n_qubits=8)[indices][:, indices]
    torch.testing.assert_close(actual, occupation_matrix(4, edges, 0.7, 2, -0.3))
    # Independent one-particle spectrum of a three-arm star, spin-degenerate.
    free = fock_matrix(fermi_hubbard_graph(4, edges), 8)
    sector = [1 << q for q in range(8)]
    levels = torch.tensor(
        [-(3**0.5)] * 2 + [0.0] * 4 + [3**0.5] * 2, dtype=torch.float64
    )
    torch.testing.assert_close(
        torch.linalg.eigvalsh(free[sector][:, sector]), levels, atol=1e-12, rtol=0
    )


def test_spectator_reverses_nonlocal_hopping_and_empty_graph_retains_onsite() -> None:
    matrix = fock_matrix(fermi_hubbard_graph(4, ((0, 3),), t=1.2), 8)
    assert matrix[128, 2] == -1.2
    assert matrix[144, 18] == 1.2  # Mode 3 lies between the hopping endpoints.
    empty = fermi_hubbard_graph(3, (), u=4, mu=0.5)
    assert empty == fermi_hubbard_chain(3, t=0, u=4, mu=0.5)
    for encoding in ENCODINGS:
        zero = encode_fermion(fermi_hubbard_graph(4, ()), n_modes=8, encoding=encoding)
        assert zero.n_modes == 8 and zero.operator.terms == ()


@pytest.mark.parametrize(
    "edges,error",
    [
        (((0, 0),), ValueError),
        (((0, 3),), ValueError),
        (((-1, 1),), ValueError),
        (((0, 1), (1, 0)), ValueError),
        (((0, 1), (0, 1)), ValueError),
        (((0,),), ValueError),
        (((0, 1, 2),), ValueError),
        (((True, 1),), TypeError),
        (((0.0, 1),), TypeError),
        (("01",), TypeError),
        ((0,), TypeError),
        ("01", TypeError),
        (None, TypeError),
    ],
)
def test_invalid_graph_is_refused(
    edges: Sequence[tuple[int, int]], error: type[Exception]
) -> None:
    for t in (0.0, 1.0):
        with pytest.raises(error):
            fermi_hubbard_graph(3, edges, t=t)


@pytest.mark.parametrize(
    "sites,error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (2.0, TypeError)],
)
def test_invalid_register(sites: int, error: type[Exception]) -> None:
    with pytest.raises(error):
        fermi_hubbard_graph(sites, ())


@pytest.mark.parametrize("name", ["t", "u", "mu"])
@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        (1j, TypeError),
        (torch.tensor(1.0), TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
    ],
)
def test_static_coefficient_boundary(
    name: str, value: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        fermi_hubbard_graph(3, (), **{name: value})
