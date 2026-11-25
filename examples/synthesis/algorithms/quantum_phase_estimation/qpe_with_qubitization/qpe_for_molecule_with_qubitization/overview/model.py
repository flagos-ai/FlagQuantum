"""Regenerate the four-bit electronic Hamiltonian used by both H2 walks."""

import itertools
import math
from dataclasses import dataclass

from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.physics import (
    PAULI_WORDS,
    PROBABILITIES,
    SELECTOR_PHASES,
)
from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.observables.fermion import FermionOperator, FermionTerm, jordan_wigner

GEOMETRY = (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735)))
SPIN_BLOCKED = (0, 2, 1, 3)
WIDTH = 4


@dataclass(frozen=True)
class Model:
    coefficients: tuple[float, ...]
    probabilities: tuple[float, ...]
    normalization: float
    full_ci: float
    hartree_fock: float
    max_reference_delta: float


def reorder_spins(operator: FermionOperator) -> FermionOperator:
    """Keep all alpha modes before beta modes and restore canonical CAR order."""
    terms = []
    for term in operator.terms:
        creation = tuple(SPIN_BLOCKED[i] for i in term.creations)
        annihilation = tuple(SPIN_BLOCKED[i] for i in term.annihilations)
        inversions = sum(
            left > right
            for modes in (creation, annihilation)
            for left, right in itertools.combinations(modes, 2)
        )
        terms.append(
            FermionTerm(
                term.coefficient * (-1) ** inversions,
                tuple(sorted(creation)),
                tuple(sorted(annihilation)),
            )
        )
    return FermionOperator(tuple(terms))


def build_model() -> Model:
    molecule = create_molecular_hamiltonian(GEOMETRY, basis="sto-3g")
    if molecule.fci_energy is None:
        raise ValueError("An exact full CI energy is required")
    active = from_native_molecule(molecule)
    observable = jordan_wigner(
        reorder_spins(fermion_hamiltonian(active)), n_modes=WIDTH
    )
    coefficients = {}
    for term in observable.terms:
        coefficient = float(term.coefficient)
        if abs(coefficient) > 1e-10:
            word = ["I"] * WIDTH
            for bit, axis in term.factors:
                word[bit] = axis.upper()
            coefficients["".join(word)] = coefficient

    # The final selector leaf mirrors its predecessor with zero probability.
    order = PAULI_WORDS[:-1]
    if coefficients.keys() != set(order):
        raise ValueError(
            "The fresh four-bit Pauli supports must match the original selector"
        )
    ordered = tuple(coefficients[word] for word in order)
    normalization = math.fsum(map(abs, ordered))
    probabilities = tuple(abs(value) / normalization for value in ordered) + (0.0,)
    if any(
        (coefficient < 0) != bool(angle)
        for coefficient, angle in zip(ordered, SELECTOR_PHASES[:-1], strict=True)
    ):
        raise ValueError("The fresh Pauli signs must match the signed selector")
    delta = max(
        abs(value - saved)
        for value, saved in zip(probabilities, PROBABILITIES, strict=True)
    )
    if delta > 1e-7:
        raise ValueError("The fresh H2 weights differ from the source tolerance")
    return Model(
        ordered,
        probabilities,
        normalization,
        molecule.fci_energy,
        molecule.hf_energy,
        delta,
    )
