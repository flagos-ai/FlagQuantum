# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/ising/hamiltonian.py and walk_operator.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/ising

"""Construct the source periodic Ising Hamiltonian and walk."""

from collections import Counter
from math import sqrt
from typing import overload

from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal
from ..core import Hamiltonian, HamiltonianTerm
from .pauli_hamiltonian_walk import (
    PauliHamiltonianWalk,
    SymbolicPauliHamiltonianWalk,
    pauli_hamiltonian_walk,
    symbolic_pauli_hamiltonian_walk,
)


def ising_hamiltonian(sites: int, j_zz: float = 1, gamma_x: float = -1) -> Hamiltonian:
    """List periodic ZZ terms first, then X terms, with source coefficient signs."""
    if type(sites) is not int or sites < 1:
        raise ValueError("Ising sites must be a positive integer")
    bonds = Counter(tuple(sorted({site, (site + 1) % sites})) for site in range(sites))
    zz = tuple(
        HamiltonianTerm(j_zz * count, "Z" * len(bond), bond)
        for bond, count in bonds.items()
        if j_zz
    )
    x = tuple(
        HamiltonianTerm(gamma_x, "X", (site,)) for site in range(sites) if gamma_x
    )
    return Hamiltonian((*zz, *x))


@overload
def ising_norm_bound(sites: int, j_zz: float, gamma_x: float) -> float: ...


@overload
def ising_norm_bound(
    sites: SymbolicInt, j_zz: SymbolicReal, gamma_x: SymbolicReal
) -> SymbolicReal: ...


def ising_norm_bound(
    sites: SymbolicInt, j_zz: SymbolicReal, gamma_x: SymbolicReal
) -> SymbolicReal:
    """Apply the source norm bound obtained by pairing ZZ and X terms."""
    squared = j_zz**2 + gamma_x**2
    if isinstance(squared, (int, float)):
        return sites * sqrt(squared)
    from sympy import sqrt as symbolic_sqrt

    return sites * symbolic_sqrt(squared)


@overload
def ising_walk(
    sites: int, epsilon: float, *, j_zz: float = 1, gamma_x: float = -1
) -> tuple[PauliHamiltonianWalk, Hamiltonian]: ...


@overload
def ising_walk(
    sites: int, epsilon: SymbolicReal, *, j_zz: float = 1, gamma_x: float = -1
) -> tuple[PauliHamiltonianWalk | SymbolicPauliHamiltonianWalk, Hamiltonian]: ...


def ising_walk(
    sites: int, epsilon: SymbolicReal, *, j_zz: float = 1, gamma_x: float = -1
) -> tuple[PauliHamiltonianWalk | SymbolicPauliHamiltonianWalk, Hamiltonian]:
    """Return the periodic Hamiltonian and its concrete walk or symbolic plan."""
    hamiltonian = ising_hamiltonian(sites, j_zz, gamma_x)
    bound = ising_norm_bound(sites, j_zz, gamma_x)
    if isinstance(epsilon, (float, int)):
        return (
            pauli_hamiltonian_walk(hamiltonian, epsilon, norm_bound=bound),
            hamiltonian,
        )
    return (
        symbolic_pauli_hamiltonian_walk(hamiltonian, epsilon, norm_bound=bound),
        hamiltonian,
    )
