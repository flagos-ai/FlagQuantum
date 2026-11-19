"""Exact cycle distances agree with exhaustive CSS logical-support searches."""

from __future__ import annotations

import itertools
import random
from typing import Literal

import pytest

from flagquantum.qec import StabilizerCode, build_memory_circuit, toric_code
from flagquantum.qec.codes import _css_helpers as helper
from flagquantum.qec.gf2 import nullspace, reduce_rows, reduce_vector

pytestmark = pytest.mark.unit


def _exhaustive(
    width: int, checks: tuple[int, ...], stabilizers: tuple[int, ...], bound: int
) -> int | None:
    pivots = reduce_rows(stabilizers)
    for weight in range(1, bound + 1):
        for positions in itertools.combinations(range(width), weight):
            vector = sum(1 << wire for wire in positions)
            if all((vector & row).bit_count() % 2 == 0 for row in checks):
                if reduce_vector(vector, pivots):
                    return weight
    return None


@pytest.mark.parametrize("size", (2, 3, 4))
def test_toric_distance_and_search_bound_match_exhaustive_supports(size: int) -> None:
    code = toric_code(size)
    for checks, stabilizers in ((code.hx, code.hz), (code.hz, code.hx)):
        commuting = tuple(helper._mask(tuple(row)) for row in checks)
        own = tuple(helper._mask(tuple(row)) for row in stabilizers)
        for bound in (size - 1, size, size + 1):
            assert helper._minimum_logical_weight(
                width=2 * size**2,
                commuting_rows=commuting,
                stabilizer_rows=own,
                bound=bound,
            ) == _exhaustive(2 * size**2, commuting, own, bound)


@pytest.mark.parametrize("seed", range(12))
def test_non_lattice_cycle_codes_match_independent_enumeration(seed: int) -> None:
    rng = random.Random(seed)
    width = 6
    endpoints = [rng.sample(range(4), 2) for _ in range(width)]
    checks = tuple(
        sum(1 << wire for wire, pair in enumerate(endpoints) if vertex in pair)
        for vertex in range(4)
    )
    cycles = nullspace(checks, width)
    # Leave exactly two logical dimensions, including disconnected multigraphs.
    stabilizers = cycles[:-2]
    graph = helper._logical_cycle_graph(width, checks, stabilizers)
    assert graph is not None
    for bound in range(1, width + 1):
        assert helper._minimum_logical_cycle_weight(graph, bound) == _exhaustive(
            width, checks, stabilizers, bound
        )
    permutation = list(range(width))
    rng.shuffle(permutation)

    def permute(row: int) -> int:
        return sum(((row >> old) & 1) << new for new, old in enumerate(permutation))

    changed_checks = tuple(permute(row) for row in reversed(checks))
    changed_stabilizers = tuple(permute(row) for row in reversed(stabilizers))
    assert helper._minimum_logical_weight(
        width=width,
        commuting_rows=changed_checks,
        stabilizer_rows=changed_stabilizers,
        bound=width,
    ) == _exhaustive(width, checks, stabilizers, width)


def test_redundant_check_representations_keep_the_existing_exact_fallback() -> None:
    code = toric_code(2)
    checks = tuple(helper._mask(tuple(row)) for row in code.hx)
    own = tuple(helper._mask(tuple(row)) for row in code.hz)
    redundant = (*checks, checks[0])
    assert helper._logical_cycle_graph(8, redundant, own) is None
    assert (
        helper._minimum_logical_weight(
            width=8, commuting_rows=redundant, stabilizer_rows=own, bound=2
        )
        == 2
    )


def test_graph_route_requires_two_logicals_and_commuting_stabilizers() -> None:
    checks = (0b11, 0b11)
    assert helper._logical_cycle_graph(2, checks, ()) is None
    assert helper._logical_cycle_graph(4, (0b1111, 0b1111), (0b0001,)) is None
    assert helper._minimum_logical_cycle_weight(((),), 3) is None


@pytest.mark.parametrize("size", (3, 5, 7, 9, 11, 13, 15))
def test_full_toric_distance_list_preserves_both_logical_qubits(size: int) -> None:
    code = toric_code(size)
    assert (code.distance, code.x_distance, code.z_distance) == (size,) * 3
    assert code.num_data_qubits == code.num_ancilla_qubits == 2 * size**2
    assert len(code.lz) == len(code.lx) == 2
    checks = tuple(helper._mask(tuple(row)) for row in code.hx)
    own = tuple(helper._mask(tuple(row)) for row in code.hz)
    assert helper._logical_cycle_graph(2 * size**2, checks, own) is not None


@pytest.mark.parametrize("size", (3, 15))
@pytest.mark.parametrize("basis", ("x", "z"))
def test_each_readout_basis_preserves_two_distinct_logical_observables(
    size: int, basis: Literal["x", "z"]
) -> None:
    code = toric_code(size)
    assert isinstance(code, StabilizerCode)
    memory = build_memory_circuit(code, rounds=3, readout_basis=basis)
    observables = memory.observables.observables
    assert memory.readout_basis == basis
    assert tuple(item.index for item in observables) == (0, 1)
    assert observables[0].pauli != observables[1].pauli
    assert all(item.pauli in code.logical_observables for item in observables)
    assert all(bool(item.pauli.x_qubits) == (basis == "x") for item in observables)
