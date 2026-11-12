"""Discover diagonal Pauli symmetries and reduce explicitly selected sectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...core.algebra.pauli import PauliOperator, PauliTerm, count
from ...core.ir import Instruction
from .clifford import conjugate_pauli, symplectic_masks
from .encoding import EncodedOperator


@dataclass(frozen=True, slots=True)
class ReductionMap:
    """Affine binary embedding from reduced basis bits into encoded full bits.

    Each row is a bit mask in logical qubit numbering, independent of integer
    statevector significance. Pivots have been eliminated from every other row.
    """

    n_qubits: int
    pivots: tuple[int, ...]
    rows: tuple[int, ...]
    values: tuple[int, ...]

    @property
    def free_qubits(self) -> tuple[int, ...]:
        return tuple(q for q in range(self.n_qubits) if q not in self.pivots)

    def full_bits(self, reduced_bits: Sequence[int]) -> tuple[int, ...]:
        """Embed one reduced basis state into the selected original sector."""
        free = self.free_qubits
        if len(reduced_bits) != len(free):
            raise ValueError("reduced bits do not match the remaining register")
        bits = [0] * self.n_qubits
        for q, value in zip(free, reduced_bits, strict=True):
            bit = count(value, "basis bit")
            if bit > 1:
                raise ValueError("basis bit must be binary")
            bits[q] = bit
        for pivot, row, value in zip(self.pivots, self.rows, self.values, strict=True):
            bits[pivot] = value ^ (sum(bits[q] for q in free if row & (1 << q)) % 2)
        return tuple(bits)

    def lift_state(self, state: torch.Tensor) -> torch.Tensor:
        """Embed a reduced vector; storage grows as 2**n_qubits, explicitly."""
        free = self.free_qubits
        if state.ndim != 1 or state.numel() != 2 ** len(free):
            raise ValueError("state shape does not match the reduced register")
        if self.n_qubits > 20:
            raise ValueError("dense state embedding is limited to 20 qubits")
        indices = []
        for index in range(state.numel()):
            bits = tuple((index >> (len(free) - q - 1)) & 1 for q in range(len(free)))
            full = self.full_bits(bits)
            indices.append(
                sum(bit << (self.n_qubits - q - 1) for q, bit in enumerate(full))
            )
        result = state.new_zeros(2**self.n_qubits)
        return result.index_copy(0, torch.tensor(indices, device=state.device), state)


@dataclass(frozen=True, slots=True)
class ReducedOperator:
    """A sector operator and an explicit embedding, including zero-qubit sectors."""

    operator: PauliOperator
    mapping: ReductionMap

    @property
    def n_qubits(self) -> int:
        return len(self.mapping.free_qubits)


def _conjugate_cx(operator: PauliOperator, control: int, target: int) -> PauliOperator:
    return conjugate_pauli(operator, Instruction("cx", (control, target)))


def reduce_z_parities(
    encoded: EncodedOperator,
    *,
    generators: Sequence[Sequence[int]],
    eigenvalues: Sequence[int],
) -> ReducedOperator:
    """Restrict commuting encoded Z parities to explicitly selected eigenvalues.

    Generators are lists of encoded qubit indices, not occupation-mode labels.
    They must be independent. This supports diagonal parity constraints only;
    general stabilizer tapering and transformed ansatzes are not implied.
    """
    if len(generators) != len(eigenvalues):
        raise ValueError("one eigenvalue is required for every generator")
    rows: list[int] = []
    values: list[int] = []
    for generator, eigenvalue in zip(generators, eigenvalues, strict=True):
        qubits = tuple(count(q, "generator qubit") for q in generator)
        if (
            not qubits
            or len(set(qubits)) != len(qubits)
            or any(q >= encoded.n_qubits for q in qubits)
        ):
            raise ValueError("generator requires distinct in-register qubits")
        if isinstance(eigenvalue, bool) or eigenvalue not in (-1, 1):
            raise ValueError("sector eigenvalues must be -1 or +1")
        rows.append(sum(1 << q for q in qubits))
        values.append(int(eigenvalue == -1))
    pivots: list[int] = []
    for column in range(encoded.n_qubits):
        rank = len(pivots)
        selected = next(
            (i for i in range(rank, len(rows)) if rows[i] & (1 << column)), None
        )
        if selected is None:
            continue
        rows[rank], rows[selected] = rows[selected], rows[rank]
        values[rank], values[selected] = values[selected], values[rank]
        for i in range(len(rows)):
            if i != rank and rows[i] & (1 << column):
                rows[i] ^= rows[rank]
                values[i] ^= values[rank]
        pivots.append(column)
    if len(pivots) != len(rows):
        raise ValueError("parity generators must be independent and consistent")
    mapping = ReductionMap(encoded.n_qubits, tuple(pivots), tuple(rows), tuple(values))
    transformed = encoded.operator
    free = mapping.free_qubits
    for pivot, row in zip(pivots, rows, strict=True):
        for q in free:
            if row & (1 << q):
                transformed = _conjugate_cx(transformed, q, pivot)
    sectors = dict(zip(pivots, values, strict=True))
    positions = {q: i for i, q in enumerate(free)}
    terms: list[PauliTerm] = []
    for term in transformed.terms:
        coefficient = term.coefficient
        factors = []
        for q, axis in term.factors:
            if q in sectors:
                if axis != "z":
                    raise ValueError(
                        "operator does not preserve the requested parity sector"
                    )
                coefficient *= (-1) ** sectors[q]
            else:
                factors.append((positions[q], axis))
        terms.append(PauliTerm(coefficient, tuple(factors)))
    return ReducedOperator(PauliOperator(tuple(terms)), mapping)


def _binary_rows(
    rows: Sequence[int], width: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return canonical reduced rows and pivot columns over GF(2)."""
    work = sorted(set(rows) - {0})
    pivots: list[int] = []
    for column in range(width):
        rank = len(pivots)
        selected = next(
            (i for i in range(rank, len(work)) if work[i] & (1 << column)), None
        )
        if selected is None:
            continue
        work[rank], work[selected] = work[selected], work[rank]
        for i in range(len(work)):
            if i != rank and work[i] & (1 << column):
                work[i] ^= work[rank]
        pivots.append(column)
    return tuple(work[: len(pivots)]), tuple(pivots)


def _binary_nullspace(rows: Sequence[int], width: int) -> tuple[int, ...]:
    reduced, pivots = _binary_rows(rows, width)
    solutions = []
    for free in range(width):
        if free in pivots:
            continue
        vector = 1 << free
        for row, pivot in zip(reduced, pivots, strict=True):
            if row & (1 << free):
                vector |= 1 << pivot
        solutions.append(vector)
    return tuple(solutions)


def discover_z_parities(
    encoded: EncodedOperator,
    *,
    candidates: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Find independent commuting Z words in an explicit encoded register.

    Args:
        encoded: Native operator with its encoding and idle-register width.
        candidates: Independent encoded Z words whose product span is searched.
            None searches all Z words; an empty sequence searches only identity.

    Returns:
        A deterministic basis of nonidentity symmetries, as ascending encoded
        qubit indices. Pass it to reduce_z_parities with explicit eigenvalues.
        Particle number N fixes only total parity (-1)**N, not other sectors.

    Raises:
        TypeError: The operator or candidate indices have incorrect types.
        ValueError: Candidates are empty words, repeated, dependent or outside
            the register. Noncommuting individual candidates are allowed.

    Examples:
        >>> from flagquantum.algorithms.fermion.encoding import encode_fermion
        >>> from flagquantum.observables import annihilate, create
        >>> hop = create(0) * annihilate(1)
        >>> encoded = encode_fermion(hop + hop.dagger(), n_modes=2)
        >>> discover_z_parities(encoded)
        ((0, 1),)
    """
    if not isinstance(encoded, EncodedOperator):
        raise TypeError("encoded must be a native EncodedOperator")
    width = encoded.n_qubits
    basis = tuple(1 << q for q in range(width))
    if candidates is not None:
        masks: list[int] = []
        for candidate in candidates:
            qubits = tuple(count(q, "candidate qubit") for q in candidate)
            if (
                not qubits
                or len(set(qubits)) != len(qubits)
                or any(q >= width for q in qubits)
            ):
                raise ValueError("candidate requires distinct in-register qubits")
            masks.append(sum(1 << q for q in qubits))
        basis, _ = _binary_rows(masks, width)
        if len(basis) != len(masks):
            raise ValueError("candidate parities must be independent")
    equations = []
    for term in encoded.operator.terms:
        x, _ = symplectic_masks(term)
        equations.append(
            sum(((x & word).bit_count() % 2) << i for i, word in enumerate(basis))
        )
    solutions = []
    for combination in _binary_nullspace(equations, len(basis)):
        word = 0
        for i, candidate_mask in enumerate(basis):
            if combination & (1 << i):
                word ^= candidate_mask
        solutions.append(word)
    canonical, _ = _binary_rows(solutions, width)
    return tuple(
        tuple(q for q in range(width) if word & (1 << q)) for word in canonical
    )
