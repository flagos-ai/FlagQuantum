"""Run the tapered first-order UCCSD ansatz with a checked local objective."""

import itertools
import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.algorithms.fermion.clifford import pauli_words_commute
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.core.algebra.pauli import PauliOperator, PauliTerm
from flagquantum.observables.fermion import FermionOperator, annihilate, create

from .model import ACTIVE_ORBITALS, WIDTH
from .symmetry import Sectors, trim

BUDGET = 200
THRESHOLD = 1e-3


@dataclass(frozen=True)
class Trial:
    energy: float
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class Optimization:
    terms: int
    width: int
    excitation_terms: tuple[int, ...]
    probes: tuple[Trial, ...]
    convergence: bool
    budget_stop: bool
    state_error: float


class QueryBudgetError(Exception):
    """Bound actual local objectives at the source's iteration configuration."""


def _bare_excitation(
    source: tuple[int, ...], target: tuple[int, ...]
) -> FermionOperator:
    operator = create(source[0])
    for wire in source[1:]:
        operator *= create(wire)
    for wire in target:
        operator *= annihilate(wire)
    return (1j / math.factorial(len(source))) * (operator - operator.dagger())


def excitation_operators(sectors: Sectors) -> tuple[PauliOperator, ...]:
    allowed = tuple(
        (spin, 2 * orbital + spin)
        for spin in range(2)
        for orbital in range(1, ACTIVE_ORBITALS)
    )
    outputs = []
    for rank in (1, 2):
        for group in itertools.combinations(allowed, rank):
            source, target = zip(*group, strict=True)
            if len(set(source)) != rank or len(set(target)) != rank:
                continue
            candidate = _bare_excitation(source, target)
            pauli_op = encode_fermion(candidate, n_modes=WIDTH).operator
            preserving = PauliOperator(
                tuple(
                    term
                    for term in pauli_op.terms
                    if all(
                        pauli_words_commute(term, gen.terms[0])
                        for gen in sectors.generators
                    )
                )
            )
            reduced = trim(sectors.reduction.reduce_operator(preserving), 1e-12)
            if any(term.factors for term in reduced.terms):
                outputs.append(reduced)
    return tuple(outputs)


def append_factor(circuit: fq.Circuit, term: PauliTerm, parameter: float) -> None:
    """Emit one Pauli rotation in the source's ordered first-order product."""
    axes = term.factors
    if not axes:
        raise ValueError("identity factors do not occur in the tapered excitations")
    wires = tuple(wire for wire, _ in axes)
    for wire, axis in axes:
        if axis == "y":
            circuit.sdg(wire)
        if axis in ("x", "y"):
            circuit.h(wire)
    for control in wires[:-1]:
        circuit.cx(control, wires[-1])
    circuit.rz(wires[-1], theta=2 * parameter * term.coefficient.real)
    for control in reversed(wires[:-1]):
        circuit.cx(control, wires[-1])
    for wire, axis in reversed(axes):
        if axis in ("x", "y"):
            circuit.h(wire)
        if axis == "y":
            circuit.s(wire)


def ansatz(
    width: int,
    hf_word: int,
    operators: tuple[PauliOperator, ...],
    parameters: tuple[float, ...],
) -> fq.Circuit:
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        if hf_word & (1 << (width - wire - 1)):
            circuit.x(wire)
    for operator, parameter in zip(operators, parameters, strict=True):
        for term in operator.terms:
            append_factor(circuit, term, parameter)
    return circuit


def check_ansatz(sectors: Sectors, operators: tuple[PauliOperator, ...]) -> float:
    width = sectors.reduction.n_qubits
    for operator in operators:
        assert all(
            pauli_words_commute(a, b)
            for a, b in itertools.combinations(operator.terms, 2)
        )
    spectra = tuple(
        torch.linalg.eigh(pauli_matrix(op, n_qubits=width)) for op in operators
    )
    errors = []
    for scale in (0.17, -0.23):
        parameters = tuple(
            scale * (i + 1) / len(operators) for i in range(len(operators))
        )
        actual = ansatz(width, sectors.hf_word, operators, parameters).state().flatten()
        reference = torch.zeros(2**width, dtype=torch.complex128)
        reference[sectors.hf_word] = 1
        for (values, vectors), parameter in zip(spectra, parameters, strict=True):
            reference = vectors @ (
                torch.exp(-1j * parameter * values) * (vectors.mH @ reference)
            )
        errors.append(float((actual - reference).abs().max()))
    error = max(errors)
    assert error < 3e-12, errors
    return error


def fit(sectors: Sectors) -> Optimization:
    operators = excitation_operators(sectors)
    error = check_ansatz(sectors, operators)
    width = sectors.reduction.n_qubits
    cost = trim(sectors.reduction.operator, THRESHOLD)
    matrix = pauli_matrix(cost, n_qubits=width)
    trace: list[Trial] = []

    def evaluate(point: torch.Tensor) -> torch.Tensor:
        if len(trace) == BUDGET:
            raise QueryBudgetError
        parameters = tuple(float(value) for value in point)
        state = ansatz(width, sectors.hf_word, operators, parameters).state().flatten()
        assert isinstance(state, torch.Tensor)
        value = torch.vdot(state, matrix @ state).real
        trace.append(Trial(float(value), parameters))
        return value

    converged = stopped = False
    try:
        result = CobylaOptimizer(maxiter=BUDGET, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, torch.zeros(len(operators), dtype=torch.float64)
        )
        converged = result.converged
    except QueryBudgetError:
        stopped = True
    return Optimization(
        len(cost.terms),
        width,
        tuple(len(op.terms) for op in operators),
        tuple(trace),
        converged,
        stopped,
        error,
    )
