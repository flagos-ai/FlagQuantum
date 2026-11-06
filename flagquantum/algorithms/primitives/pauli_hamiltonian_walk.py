# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/ising/walk_operator.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/ising/walk_operator.py

"""Construct an alias-prepared walk from ordered real Pauli coefficients."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import overload

from ...compiler._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...compiler.registers.bounded_pauli_select import source_bounded_pauli_select_cost
from ...compiler.registers.prepare_reflection import (
    source_uncontrolled_prepare_reflection_cost,
)
from ...core.ir import Instruction
from ..core import Hamiltonian
from .alias_preparation import (
    alias_black_box_prepare,
    source_dense_alias_preparation_cost,
)
from .alias_sampling import AliasSamplingTable, alias_sampling_table
from .multiplexers.select_pauli_lcu import SelectPauliLCU
from .qubitization_walk import QubitizationWalk
from .select_block_encoding import SelectBlockEncoding
from .symbolic_alias_preparation import AliasPreparationShape


@overload
def prepare_precision_from_eigenphase(
    epsilon: float, terms: int, normalization: float, norm_bound: float
) -> float: ...


@overload
def prepare_precision_from_eigenphase(
    epsilon: SymbolicReal,
    terms: SymbolicReal,
    normalization: SymbolicReal,
    norm_bound: SymbolicReal,
) -> SymbolicReal: ...


def prepare_precision_from_eigenphase(
    epsilon: SymbolicReal,
    terms: SymbolicReal,
    normalization: SymbolicReal,
    norm_bound: SymbolicReal,
) -> SymbolicReal:
    """Return the source's absolute coefficient tolerance from eigenphase precision."""
    return ((epsilon * normalization) / ((1 + epsilon**2) * terms)) * (
        1 - (norm_bound / normalization) ** 2
    )


@dataclass(frozen=True)
class PauliHamiltonianWalk:
    """Concrete walk, alias table and logical Hamiltonian target ordering.

    The locked reflection can leave preparation junk nonzero. Keep those
    wires allocated after a step and account for them in any later use.
    """

    walk: QubitizationWalk
    table: AliasSamplingTable
    selector: SelectPauliLCU
    target_qubits: tuple[int, ...]
    coefficient_precision: float

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        """Price the bounded SELECT and alias reflection without emitting IR."""
        prep = source_dense_alias_preparation_cost(self.table)
        return self.walk.resource_model(
            _selector_cost(self.selector), prep, prep, adjoint=adjoint
        )


def _selector_cost(selector: SelectPauliLCU) -> ComponentCost:
    address = selector.selection_bitsize
    instruction = selector._instruction(
        tuple(range(address)),
        tuple(range(address, address + selector.target_bitsize)),
        (),
    )
    return source_bounded_pauli_select_cost(instruction)


@dataclass(frozen=True)
class SymbolicPauliHamiltonianWalk:
    """Source gate-cost plan for a concrete Hamiltonian with symbolic precision."""

    prepare: AliasPreparationShape
    selector: SelectPauliLCU
    target_qubits: tuple[int, ...]
    coefficient_precision: SymbolicReal

    def gate_cost(self) -> QECGateCost:
        """Query symbolic gate categories while the source peak is undefined."""
        select = query_qec_gate_cost(_selector_cost(self.selector))
        identity = ComponentCost("Identity", self.selector.selection_bitsize)
        reflection = source_uncontrolled_prepare_reflection_cost(
            self.selector.selection_bitsize, 0, identity, identity
        )
        return select.plus(query_qec_gate_cost(reflection), 1).plus(
            self.prepare.resource_model(), 2
        )


def symbolic_pauli_hamiltonian_walk(
    hamiltonian: Hamiltonian, epsilon: SymbolicReal, *, norm_bound: float | None = None
) -> SymbolicPauliHamiltonianWalk:
    """Apply the source shape-only PREPARE path to a symbolic precision."""
    targets, words, weights, norm = _walk_inputs(hamiltonian, norm_bound)
    normalization = sum(weights)
    precision = prepare_precision_from_eigenphase(
        epsilon, len(words), normalization, norm
    )
    shape = AliasPreparationShape.from_precision(len(words), normalization, precision)
    selector = SelectPauliLCU((len(words) - 1).bit_length(), len(targets), words)
    return SymbolicPauliHamiltonianWalk(shape, selector, targets, precision)


def pauli_hamiltonian_walk(
    hamiltonian: Hamiltonian,
    epsilon: float,
    *,
    norm_bound: float | None = None,
) -> PauliHamiltonianWalk:
    """Build a source-bounded walk from a concrete FlagQuantum Hamiltonian.

    Repeated products combine in first-occurrence order. Active target labels
    are sorted. Signs go to SELECT and magnitudes go to alias preparation.
    The automatic estimate preserves the source signed-coefficient formula.
    A supplied norm bound must be finite and strictly below the L1 norm
    to give a positive source precision.
    """
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Eigenphase precision must be positive and finite")
    targets, words, weights, norm = _walk_inputs(hamiltonian, norm_bound)
    precision = prepare_precision_from_eigenphase(
        epsilon, len(words), sum(weights), norm
    )
    table = alias_sampling_table(weights, precision=precision)
    prepare = alias_black_box_prepare(table)
    selector = SelectPauliLCU(prepare.selection_width, len(targets), words)

    def select(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return selector.make_bounded_on(
            selection=ports["selection"],
            target=ports["target"],
            workspace=work,
            classical_bits=slots,
        ).emit()

    block = SelectBlockEncoding(
        prepare.selection_layout, (("target", len(targets)),), select, prepare
    )
    return PauliHamiltonianWalk(
        QubitizationWalk(block), table, selector, targets, precision
    )


def _walk_inputs(
    hamiltonian: Hamiltonian, norm_bound: float | None
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[float, ...], float]:
    combined: dict[tuple[tuple[int, str], ...], float] = {}
    for term in hamiltonian.terms:
        coefficient = term.coefficient
        if not isinstance(coefficient, (float, int, complex)):
            raise ValueError("Walk coefficients must be concrete real scalars")
        if coefficient.imag or not isfinite(coefficient.real):
            raise ValueError("Walk coefficients must be concrete finite real scalars")
        key = tuple(sorted(term.ops))
        combined[key] = combined.get(key, 0.0) + coefficient.real
    terms = tuple((dict(ops), weight) for ops, weight in combined.items() if weight)
    if len(terms) < 2:
        raise ValueError("A source alias walk requires at least two nonzero products")

    targets = tuple(sorted({qubit for ops, _ in terms for qubit in ops}))
    weights = tuple(abs(weight) for _, weight in terms)
    normalization = sum(weights)
    if not isfinite(normalization):
        raise ValueError("Walk normalization must be finite")
    if norm_bound is None:
        signed_sum = sum(weight for _, weight in terms)
        anticommute = sum(
            abs(weight * other_weight)
            for i, (ops, weight) in enumerate(terms)
            for other, other_weight in terms[:i]
            if sum(ops[q] != other[q] for q in ops.keys() & other.keys()) % 2
        )
        squared = signed_sum**2 - 2 * anticommute
        if squared < 0:
            raise ValueError(
                "Source norm estimate is nonreal; supply a valid norm bound"
            )
        norm_bound = sqrt(squared)
    if not isfinite(norm_bound) or not 0 <= norm_bound < normalization:
        raise ValueError("Norm bound must be finite and below the L1 normalization")

    words = tuple(
        ("-" if weight < 0 else "") + "".join(ops.get(q, "i").upper() for q in targets)
        for ops, weight in terms
    )
    return targets, words, weights, norm_bound
