"""Symbolic restriction to explicitly supplied commuting Pauli symmetries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.algebra.pauli import PauliOperator, PauliTerm, count, pauli
from ...core.ir import Instruction
from .clifford import conjugate_pauli, pauli_words_commute, symplectic_masks
from .symmetry import ReductionMap


def _checked_operator(operator: PauliOperator, width: int) -> PauliOperator:
    if not isinstance(operator, PauliOperator):
        raise TypeError("operator and generators must be native PauliOperator values")
    if any(q >= width for term in operator.terms for q, _ in term.factors):
        raise ValueError("Pauli operator acts outside the declared register")
    return operator


def _restrict(operator: PauliOperator, mapping: ReductionMap) -> PauliOperator:
    sectors = dict(zip(mapping.pivots, mapping.values, strict=True))
    positions = {q: i for i, q in enumerate(mapping.free_qubits)}
    terms: list[PauliTerm] = []
    for term in operator.terms:
        coefficient = term.coefficient
        factors = []
        for q, axis in term.factors:
            if q in sectors:
                if axis != "z":
                    raise ValueError(
                        "operator does not preserve the selected stabilizer sector"
                    )
                coefficient *= (-1) ** sectors[q]
            else:
                factors.append((positions[q], axis))
        terms.append(PauliTerm(coefficient, tuple(factors)))
    return PauliOperator(tuple(terms))


@dataclass(frozen=True, slots=True)
class StabilizerReduction:
    """A reduced operator, Clifford frame and diagonal-frame sector embedding.

    ``clifford`` describes C in native instruction application order, and
    ``mapping`` embeds into its diagonal frame. The original sector is embedded
    by C dagger mapping, not by mapping alone. This extra Clifford is essential:
    an X/Y eigenstate cannot be expressed by the existing affine basis-bit map.
    Free qubits are ordered by mapping.free_qubits in that diagonal frame.
    """

    operator: PauliOperator
    mapping: ReductionMap
    clifford: tuple[Instruction, ...]

    @property
    def n_qubits(self) -> int:
        return len(self.mapping.free_qubits)

    def reduce_operator(self, operator: PauliOperator) -> PauliOperator:
        """Restrict another preserving operator using exactly the same frame.

        Reject a nonpreserving operator instead of silently projecting it out.
        Complex coefficients are retained; measurement uses to_observable.
        """
        transformed = _checked_operator(operator, self.mapping.n_qubits)
        for gate in self.clifford:
            transformed = conjugate_pauli(transformed, gate)
        return _restrict(transformed, self.mapping)


def _validated_generators(
    operator: PauliOperator,
    width: int,
    generators: Sequence[PauliOperator],
    eigenvalues: Sequence[int],
) -> tuple[list[PauliOperator], list[int]]:
    if len(generators) != len(eigenvalues):
        raise ValueError("one eigenvalue is required for every stabilizer generator")
    words: list[PauliOperator] = []
    values: list[int] = []
    basis: dict[int, int] = {}
    for generator, eigenvalue in zip(generators, eigenvalues, strict=True):
        _checked_operator(generator, width)
        if len(generator.terms) != 1:
            raise ValueError("a stabilizer generator must be one signed Pauli word")
        term = generator.terms[0]
        if not term.factors or term.coefficient not in (-1, 1):
            raise ValueError(
                "a stabilizer generator must be nonidentity with coefficient -1 or +1"
            )
        value = count(eigenvalue, "sector eigenvalue", minimum=-1)
        if value not in (-1, 1):
            raise ValueError("sector eigenvalues must be -1 or +1")
        if any(not pauli_words_commute(term, other.terms[0]) for other in words):
            raise ValueError("stabilizer generators must commute with one another")
        x, z = symplectic_masks(term)
        vector = x | (z << width)
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = vector
                break
            vector ^= basis[pivot]
        if not vector:
            raise ValueError("stabilizer generators must be independent")
        if any(not pauli_words_commute(term, other) for other in operator.terms):
            raise ValueError(
                "operator does not preserve the selected stabilizer sector"
            )
        words.append(generator)
        values.append(value)
    return words, values


def reduce_stabilizers(
    operator: PauliOperator,
    *,
    n_qubits: int,
    generators: Sequence[PauliOperator],
    eigenvalues: Sequence[int],
) -> StabilizerReduction:
    """Restrict a native Pauli operator to an explicit joint symmetry sector.

    Each generator must be a single nonidentity Pauli word with coefficient
    +1 or -1. Generators must be independent, commute pairwise and commute with
    the operator. Eigenvalues refer to the signed generators as supplied.
    No automatic symmetry discovery or change of fermion encoding is performed.

    The Clifford uses H, S dagger and CX, with qubit zero most significant in
    statevectors. Empty generators give the identity reduction; a fully fixed
    sector may leave zero qubits. Symbolic work never allocates a dense matrix.
    See lift_stabilizer_state for bounded state embedding in the original frame.

    Args:
        operator: Operator whose complex coefficients are retained exactly.
        n_qubits: Full register width, including idle qubits; may be zero.
        generators: Independent commuting signed native Pauli words.
        eigenvalues: One integer -1 or +1 per generator.

    Returns:
        Reduced operator and an explicit native Clifford/diagonal embedding.

    Raises:
        TypeError: Inputs have the wrong native type or a count is not integer.
        ValueError: A word, width, symmetry or selected sector is invalid.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> from flagquantum.core.algebra.pauli import pauli
        >>> from flagquantum.algorithms.fermion.stabilizer import reduce_stabilizers
        >>> from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
        >>> x = pauli(factors=((0, "x"),))
        >>> reduced = reduce_stabilizers(x, n_qubits=1, generators=[x], eigenvalues=[1])
        >>> reduced.n_qubits
        0
        >>> state = lift_stabilizer_state(reduced, torch.ones(1))
        >>> circuit = fq.Circuit(1, dtype=torch.complex128, inputs=state)
        >>> [round(p, 6) for p in circuit.state().abs().square().reshape(-1).tolist()]
        [0.5, 0.5]
    """
    width = count(n_qubits, "n_qubits")
    _checked_operator(operator, width)
    words, values = _validated_generators(operator, width, generators, eigenvalues)
    transformed = operator
    gates: list[Instruction] = []
    pivots: list[int] = []
    bits: list[int] = []

    def apply(gate: Instruction) -> None:
        nonlocal transformed, words
        gates.append(gate)
        transformed = conjugate_pauli(transformed, gate)
        words = [conjugate_pauli(word, gate) for word in words]

    for index, value in enumerate(values):
        effective = words[index]
        factors = dict(effective.terms[0].factors)
        for pivot, bit in zip(pivots, bits, strict=True):
            if pivot in factors:
                effective = effective * pauli(factors=((pivot, "z"),))
                value *= (-1) ** bit
        term = effective.terms[0]
        # Independence guarantees a free support; earlier pivots commute with
        # every remaining generator and therefore carry only I or Z.
        pivot = term.factors[0][0]
        for qubit, axis in term.factors:
            if axis == "y":
                apply(Instruction("sdg", (qubit,)))
            if axis != "z":
                apply(Instruction("h", (qubit,)))
        for qubit, _ in term.factors:
            if qubit != pivot:
                apply(Instruction("cx", (qubit, pivot)))
        pivots.append(pivot)
        bits.append(int(value / term.coefficient == -1))
    mapping = ReductionMap(
        width, tuple(pivots), tuple(1 << q for q in pivots), tuple(bits)
    )
    return StabilizerReduction(_restrict(transformed, mapping), mapping, tuple(gates))
