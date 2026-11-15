# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/ising/hamiltonian.py and walk_operator.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/ising

"""Construct the source periodic Ising Hamiltonian and walk."""

from typing import overload

from ...compiler._component_trace.symbolic_math import SymbolicReal
from ..core import Hamiltonian
from .ising_hamiltonian import ising_hamiltonian as ising_hamiltonian
from .ising_hamiltonian import ising_norm_bound as ising_norm_bound
from .pauli_hamiltonian_walk import (
    PauliHamiltonianWalk,
    SymbolicPauliHamiltonianWalk,
    pauli_hamiltonian_walk,
    symbolic_pauli_hamiltonian_walk,
)


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
