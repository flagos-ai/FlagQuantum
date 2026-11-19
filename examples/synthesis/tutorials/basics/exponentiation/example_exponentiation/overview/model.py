"""Compute fresh STO-3G integrals and the ordered active-space Pauli terms."""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import flagquantum as fq
from examples.synthesis.pauli_evolution import Term, load_terms
from flagquantum.algorithms.chemistry_domain.problem import (
    fermion_hamiltonian,
    from_native_molecule,
)
from flagquantum.algorithms.molecular import create_molecular_hamiltonian
from flagquantum.observables.fermion import FermionOperator, FermionTerm, jordan_wigner

GEOMETRY = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.586, 0.757)),
    ("H", (0.0, 0.586, -0.757)),
)
BASIS = "sto-3g"
ACTIVE_SPATIAL = 6
ACTIVE_ELECTRONS = 8
WIDTH = 12
TERM_COUNT = 551
COEFFICIENT_CUTOFF = 1e-10


@dataclass(frozen=True)
class Model:
    hamiltonian: fq.Observable
    terms: tuple[Term, ...]
    rhf_energy: float
    frozen_orbitals: int
    coefficient_difference: float


def spin_block_order(
    fermion: FermionOperator, orbital_signs: tuple[int, ...]
) -> FermionOperator:
    """Change the interleaved modes to all-alpha then all-beta, with CAR signs."""
    alpha = tuple(2 * orbital for orbital in range(ACTIVE_SPATIAL))
    beta = tuple(mode + 1 for mode in alpha)
    permutation = {mode: index for index, mode in enumerate((*alpha, *beta))}
    terms = []
    for item in fermion.terms:
        creations = tuple(permutation[mode] for mode in item.creations)
        annihilations = tuple(permutation[mode] for mode in item.annihilations)
        inversions = sum(
            left > right
            for modes in (creations, annihilations)
            for left, right in itertools.combinations(modes, 2)
        )
        gauge = math.prod(
            orbital_signs[mode // 2] for mode in (*item.creations, *item.annihilations)
        )
        terms.append(
            FermionTerm(
                item.coefficient * (-1) ** inversions * gauge,
                tuple(sorted(creations)),
                tuple(sorted(annihilations)),
            )
        )
    return FermionOperator(tuple(terms))


def build_model() -> Model:
    molecule = create_molecular_hamiltonian(
        GEOMETRY,
        basis=BASIS,
        n_active_electrons=ACTIVE_ELECTRONS,
        n_active_orbitals=ACTIVE_SPATIAL,
    )
    frozen = molecule.n_frozen_core_orbitals
    coeff = molecule.hartree_fock.coefficients[:, frozen:]
    # Canonicalize each real orbital by its largest-magnitude AO coefficient.
    # The original RHF source uses this convention.
    signs = tuple(
        1 if float(column[column.abs().argmax()]) > 0 else -1 for column in coeff.T
    )
    active = from_native_molecule(molecule)
    ordered = spin_block_order(fermion_hamiltonian(active), signs)
    encoded = jordan_wigner(ordered, n_modes=WIDTH)
    fresh = {
        tuple((lane, axis.upper()) for lane, axis in term.factors): float(
            term.coefficient
        )
        for term in encoded.terms
        if abs(float(term.coefficient)) > COEFFICIENT_CUTOFF
    }
    width, reference = load_terms(
        Path(__file__).resolve().parents[1] / "hamiltonian.json"
    )
    ordering = tuple(tuple(sorted(support)) for _, support in reference)
    if width != WIDTH or len(fresh) != TERM_COUNT or set(ordering) != set(fresh):
        raise ValueError(
            "The regenerated molecular Pauli supports differ from the source"
        )
    # The retained sequence specifies the product order. Every coefficient
    # sent to synthesis comes from the fresh native electronic-structure run.
    terms = tuple((fresh[support], support) for support in ordering)
    delta = max(
        abs(generated - saved)
        for (generated, _), (saved, _) in zip(terms, reference, strict=True)
    )
    if delta >= 1e-6:
        raise ValueError(
            "The fresh molecular coefficients do not match the source tolerance"
        )

    axes = {"X": fq.X, "Y": fq.Y, "Z": fq.Z}
    hamiltonian = 0.0 * fq.I()
    for value, support in terms:
        term = fq.I()
        for lane, axis in support:
            term = term @ axes[axis](lane)
        hamiltonian += value * term
    return Model(hamiltonian, terms, molecule.hf_energy, frozen, delta)
