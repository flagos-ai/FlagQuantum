# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/ising/hamiltonian.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/ising

"""Periodic Ising Pauli terms, ordered coefficients and norm bound."""

from collections.abc import Sequence
from math import pi, sqrt
from typing import overload

from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal
from ..core import Hamiltonian, HamiltonianTerm


def periodic_ising_terms(
    qubits: Sequence[int], j_zz: float = 1, gamma_x: float = -1
) -> tuple[tuple[HamiltonianTerm, ...], tuple[HamiltonianTerm, ...]]:
    """Return the raw periodic ZZ and X terms in lattice input order."""
    bonds = (
        tuple(sorted({qubit, qubits[(site + 1) % len(qubits)]}))
        for site, qubit in enumerate(qubits)
    )
    zz = tuple(HamiltonianTerm(j_zz, "Z" * len(bond), bond) for bond in bonds)
    x = tuple(HamiltonianTerm(gamma_x, "X", (qubit,)) for qubit in qubits)
    return zz, x


def _ising_weights(
    qubits: Sequence[int], j_zz: float, gamma_x: float
) -> dict[tuple[tuple[int, str], ...], float]:
    zz, x = periodic_ising_terms(qubits, j_zz, gamma_x)
    weights: dict[tuple[tuple[int, str], ...], float] = {}
    for terms, coefficient in ((zz, j_zz), (x, gamma_x)):
        for term in terms:
            weights[term.ops] = weights.get(term.ops, 0.0) + coefficient
    return {ops: coefficient for ops, coefficient in weights.items() if coefficient}


def ising_hamiltonian(
    sites: int | Sequence[int], j_zz: float = 1, gamma_x: float = -1
) -> Hamiltonian:
    """Build the merged Hamiltonian over a site count or ordered wire labels."""
    if isinstance(sites, int):
        if type(sites) is not int or sites < 1:
            raise ValueError("Ising sites must be a positive integer")
        sites = range(sites)
    return Hamiltonian(
        HamiltonianTerm(
            coefficient,
            "".join(pauli.upper() for _, pauli in ops),
            tuple(qubit for qubit, _ in ops),
        )
        for ops, coefficient in _ising_weights(sites, j_zz, gamma_x).items()
    )


def ising_lcu_coefficients(
    sites: int, j_zz: float = pi / 3, gamma_x: float = pi / 7
) -> tuple[float, ...]:
    """Return signed coefficients in the merged Hamiltonian's SELECT order."""
    if type(sites) is not int or sites < 0:
        raise ValueError("Ising sites must be a nonnegative integer")
    return tuple(_ising_weights(range(sites), j_zz, gamma_x).values())


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
