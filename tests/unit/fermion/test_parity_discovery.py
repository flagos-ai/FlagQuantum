"""Discover diagonal symmetries against exhaustive independent commutators."""

import itertools
import random

import pytest
import torch

from flagquantum.algorithms.fermion.encoding import (
    ENCODINGS,
    EncodedOperator,
    encode_fermion,
)
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.symmetry import (
    discover_z_parities,
    reduce_z_parities,
)
from flagquantum.core.algebra.pauli import PauliOperator, pauli
from flagquantum.observables import annihilate, create, number

pytestmark = pytest.mark.unit


def _span(words: tuple[tuple[int, ...], ...]) -> set[int]:
    values = {0}
    for word in words:
        mask = sum(1 << q for q in word)
        values |= {value ^ mask for value in values.copy()}
    return values


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_complete_diagonal_commutant_against_dense_enumeration(width: int) -> None:
    rng = random.Random(42 + width)
    for _ in range(5):
        operator = PauliOperator()
        for _ in range(6):
            axes = [rng.choice("ixyz") for _ in range(width)]
            operator += pauli(rng.choice([1, -2, 0.25j]), tuple(enumerate(axes)))
        encoded = EncodedOperator(operator, "jordan_wigner", width)
        actual = discover_z_parities(encoded)
        matrix = pauli_matrix(operator, n_qubits=width)
        expected = set()
        for mask in range(1 << width):
            diagonal = torch.tensor(
                [
                    (-1)
                    ** sum(
                        ((index >> (width - 1 - q)) & 1)
                        for q in range(width)
                        if mask & (1 << q)
                    )
                    for index in range(1 << width)
                ],
                dtype=torch.complex128,
            )
            if torch.equal(diagonal[:, None] * matrix, matrix * diagonal[None, :]):
                expected.add(mask)
        assert _span(actual) == expected
        assert len(_span(actual)) == 2 ** len(actual)
        candidates = tuple((q,) for q in range(width) if q % 2 == 0)
        restricted = discover_z_parities(encoded, candidates=candidates)
        assert _span(restricted) == expected & _span(candidates)


def test_candidate_products_can_commute_when_individual_words_do_not() -> None:
    encoded = EncodedOperator(pauli(factors=((0, "x"), (1, "y"))), "jordan_wigner", 3)
    assert discover_z_parities(encoded, candidates=[(0,), (1,)]) == ((0, 1),)
    assert discover_z_parities(encoded, candidates=[]) == ()
    assert discover_z_parities(encoded, candidates=[(0,)]) == ()
    assert discover_z_parities(encoded) == ((0, 1), (2,))
    assert discover_z_parities(encoded, candidates=[(1,), (0,)]) == ((0, 1),)
    assert discover_z_parities(encoded, candidates=[(0, 1), (1,)]) == ((0, 1),)


def test_zero_identity_and_exact_cancellation_retain_idle_modes() -> None:
    x = pauli(factors=((0, "x"),))
    for operator in [PauliOperator(), pauli(2j), x - x]:
        assert discover_z_parities(EncodedOperator(operator, "parity", 3)) == (
            (0,),
            (1,),
            (2,),
        )
    assert discover_z_parities(EncodedOperator(1e-20 * x, "parity", 1)) == ()


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_discovered_sectors_intertwine_in_every_encoding(encoding: str) -> None:
    # One connected hopping component plus an independently occupied mode.
    hop = create(0) * annihilate(2)
    encoded = encode_fermion(
        hop + hop.dagger() + 0.3 * number(1), n_modes=3, encoding=encoding
    )
    generators = discover_z_parities(encoded)
    assert len(generators) == 2
    full = pauli_matrix(encoded.operator, n_qubits=3)
    for values in itertools.product((-1, 1), repeat=len(generators)):
        reduced = reduce_z_parities(encoded, generators=generators, eigenvalues=values)
        embedding = torch.stack(
            [
                reduced.mapping.lift_state(column)
                for column in torch.eye(2**reduced.n_qubits, dtype=torch.complex128)
            ],
            dim=1,
        )
        small = pauli_matrix(reduced.operator, n_qubits=reduced.n_qubits)
        torch.testing.assert_close(
            full @ embedding, embedding @ small, atol=1e-12, rtol=0
        )
        for word, value in zip(generators, values, strict=True):
            generator = pauli_matrix(
                pauli(factors=tuple((q, "z") for q in word)), n_qubits=3
            )
            torch.testing.assert_close(
                generator @ embedding, value * embedding, atol=0, rtol=0
            )


def test_parity_does_not_imply_particle_number_conservation() -> None:
    pair = create(0) * create(1)
    pairing = encode_fermion(pair + pair.dagger(), n_modes=2)
    assert discover_z_parities(pairing) == ((0, 1),)
    total = encode_fermion(number(0) + number(1), n_modes=2).operator
    assert pairing.operator * total != total * pairing.operator
    source = create(0) + annihilate(0)
    assert discover_z_parities(encode_fermion(source, n_modes=1)) == ()


@pytest.mark.parametrize(
    "candidates", [[()], [(0, 0)], [(2,)], [(-1,)], [(0,), (0,)], [(0,), (1,), (0, 1)]]
)
def test_invalid_candidates_are_refused(candidates: list[tuple[int, ...]]) -> None:
    encoded = encode_fermion(number(0), n_modes=2)
    with pytest.raises(ValueError):
        discover_z_parities(encoded, candidates=candidates)


@pytest.mark.parametrize("index", [True, 0.5, "0"])
def test_candidate_types_are_not_coerced(index: object) -> None:
    with pytest.raises(TypeError):
        discover_z_parities(encode_fermion(number(0), n_modes=2), candidates=[(index,)])


def test_wrong_operator_type_is_refused() -> None:
    with pytest.raises(TypeError, match="EncodedOperator"):
        discover_z_parities(pauli())


def test_wide_symbolic_discovery_needs_no_fock_space() -> None:
    width = 128
    operator = PauliOperator()
    for q in range(width - 1):
        operator += pauli(factors=((q, "x"), (q + 1, "x")))
    encoded = EncodedOperator(operator, "jordan_wigner", width)
    assert discover_z_parities(encoded) == (tuple(range(width)),)


def test_equivalent_candidate_bases_give_identical_canonical_output() -> None:
    encoded = EncodedOperator(pauli(factors=((0, "x"), (1, "x"))), "jordan_wigner", 4)
    expected = discover_z_parities(encoded)
    for candidates in [((3,), (1,), (0,), (2,)), ((0, 1), (1, 2), (2, 3), (3,))]:
        assert discover_z_parities(encoded, candidates=candidates) == expected
