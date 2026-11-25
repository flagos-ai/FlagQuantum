"""Discover the active LiH diagonal symmetries and verify the compiled sectors."""

import itertools
import math
from dataclasses import dataclass

import torch

from flagquantum.algorithms.fermion.clifford import (
    pauli_words_commute,
    symplectic_masks,
)
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.stabilizer import (
    StabilizerReduction,
    reduce_stabilizers,
)
from flagquantum.core.algebra.pauli import PauliOperator, pauli
from flagquantum.observables.fermion import FermionOperator

from .model import WIDTH

HF_WORD = (1 << (WIDTH - 1)) | (1 << (WIDTH - 2))


@dataclass(frozen=True)
class Sectors:
    hamiltonian: PauliOperator
    generators: tuple[PauliOperator, ...]
    dual_x: tuple[PauliOperator, ...]
    block: PauliOperator
    minima: tuple[tuple[tuple[int, ...], float], ...]
    hf_sector: tuple[int, ...]
    hf_word: int
    reduction: StabilizerReduction
    spectral_error: float


def trim(operator: PauliOperator, threshold: float) -> PauliOperator:
    return PauliOperator(
        tuple(t for t in operator.terms if abs(t.coefficient) > threshold)
    )


def diagonal_masks(operator: PauliOperator) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Enumerate the diagonal null space over GF(2) for the 10-wire source."""
    active = {symplectic_masks(term)[0] for term in operator.terms}
    rows: dict[int, int] = {}
    for number in range(1, 2**WIDTH):
        if any((number & flipped).bit_count() % 2 for flipped in active):
            continue
        remainder = number
        for pivot, row in sorted(rows.items()):
            if remainder & (1 << pivot):
                remainder ^= row
        if remainder:
            pivot = (remainder & -remainder).bit_length() - 1
            rows = {
                column: row ^ remainder if (row >> pivot) & 1 else row
                for column, row in rows.items()
            }
            rows[pivot] = remainder
    return tuple(rows[pivot] for pivot in sorted(rows)), tuple(sorted(rows))


def _z_eigenvalues(masks: tuple[int, ...], word: int) -> tuple[int, ...]:
    physical_bits = int(f"{word:0{WIDTH}b}"[::-1], 2)
    return tuple((-1) ** ((physical_bits & mask).bit_count() % 2) for mask in masks)


def analyze(fermion: FermionOperator) -> Sectors:
    h = trim(encode_fermion(fermion, n_modes=WIDTH).operator, 1e-13)
    masks, pivots = diagonal_masks(h)
    generators = tuple(
        pauli(factors=tuple((i, "z") for i in range(WIDTH) if (mask >> i) & 1))
        for mask in masks
    )
    dual_x = tuple(pauli(factors=((i, "x"),)) for i in pivots)
    transform = pauli()
    for i, (gen, x_op) in enumerate(zip(generators, dual_x, strict=True)):
        assert all(pauli_words_commute(gen.terms[0], term) for term in h.terms)
        for j, other in enumerate(generators):
            assert pauli_words_commute(x_op.terms[0], other.terms[0]) == (i != j)
        transform = transform * ((gen + x_op) * (1 / math.sqrt(2)))
    for gen, x_op in zip(generators, dual_x, strict=True):
        difference = trim(transform * gen * transform - x_op, 1e-12)
        assert not difference.terms
    block = trim(transform * h * transform, 1e-12)
    expected_sector = _z_eigenvalues(masks, HF_WORD)
    chosen = reduce_stabilizers(
        h, n_qubits=WIDTH, generators=generators, eigenvalues=expected_sector
    )
    h_dense = pauli_matrix(h, n_qubits=WIDTH)
    maxima = []
    minima = []
    for sector in itertools.product((1, -1), repeat=len(masks)):
        reduction = reduce_stabilizers(
            h, n_qubits=WIDTH, generators=generators, eigenvalues=sector
        )
        words = torch.tensor(
            [word for word in range(2**WIDTH) if _z_eigenvalues(masks, word) == sector],
            dtype=torch.long,
        )
        direct = h_dense.index_select(0, words).index_select(1, words)
        native = pauli_matrix(reduction.operator, n_qubits=reduction.n_qubits)
        reference = torch.linalg.eigvalsh(direct)
        spectrum = torch.linalg.eigvalsh(native)
        maxima.append(float((reference - spectrum).abs().max()))
        minima.append((sector, float(spectrum[0])))
    error = max(maxima)
    assert error < 2e-11
    word = HF_WORD
    for gate in chosen.clifford:
        assert gate.name == "cx"
        left, right = gate.wires
        if (word >> (WIDTH - left - 1)) & 1:
            word ^= 1 << (WIDTH - right - 1)
    reduced_hf = 0
    for index in chosen.mapping.free_qubits:
        reduced_hf = 2 * reduced_hf + ((word >> (WIDTH - index - 1)) & 1)
    return Sectors(
        h,
        generators,
        dual_x,
        block,
        tuple(minima),
        expected_sector,
        reduced_hf,
        chosen,
        error,
    )
