"""Spin-chain Hamiltonian models."""

from __future__ import annotations

from importlib import import_module as _import_public_namespace

from flagquantum.algorithms.primitives.hamiltonian import (
    Hamiltonian,
    HamiltonianTerm,
    pauli_term,
)


def transverse_field_ising(
    n_qubits: int,
    *,
    coupling: float = 1.0,
    field: float = 1.0,
    periodic: bool = False,
) -> Hamiltonian:
    """Build ``-coupling * ZZ - field * X`` transverse-field Ising Hamiltonian."""

    terms: list[HamiltonianTerm] = []
    for qubit in range(n_qubits - 1):
        terms.append(pauli_term(-coupling, "ZZ", (qubit, qubit + 1)))
    if periodic and n_qubits > 2:
        terms.append(pauli_term(-coupling, "ZZ", (n_qubits - 1, 0)))
    for qubit in range(n_qubits):
        terms.append(pauli_term(-field, "X", (qubit,)))
    return Hamiltonian(terms)


def zz_chain_hamiltonian(
    n_qubits: int,
    *,
    coupling: float = 1.0,
    field: float = 0.0,
    periodic: bool = False,
) -> Hamiltonian:
    """Build a nearest-neighbor ``coupling * ZZ + field * Z`` Hamiltonian."""

    terms: list[HamiltonianTerm] = []
    for qubit in range(n_qubits - 1):
        terms.append(pauli_term(coupling, "ZZ", (qubit, qubit + 1)))
    if periodic and n_qubits > 2:
        terms.append(pauli_term(coupling, "ZZ", (n_qubits - 1, 0)))
    for qubit in range(n_qubits):
        if field:
            terms.append(pauli_term(field, "Z", (qubit,)))
    return Hamiltonian(terms)


def heisenberg_chain_hamiltonian(
    n_qubits: int, *, anisotropy: float = 1.0, field: float = 0.0
) -> Hamiltonian:
    """Return the open-boundary XX + YY + anisotropy ZZ Hamiltonian."""

    if n_qubits < 2:
        raise ValueError("Heisenberg chains require at least two qubits")
    terms: list[HamiltonianTerm] = []
    for left in range(n_qubits - 1):
        terms.extend(
            (
                pauli_term(1.0, "XX", (left, left + 1)),
                pauli_term(1.0, "YY", (left, left + 1)),
                pauli_term(anisotropy, "ZZ", (left, left + 1)),
            )
        )
    if field:
        terms.extend(pauli_term(field, "Z", (qubit,)) for qubit in range(n_qubits))
    return Hamiltonian(terms)


__all__ = (
    "transverse_field_ising",
    "zz_chain_hamiltonian",
    "heisenberg_chain_hamiltonian",
)

for _identity in (
    transverse_field_ising,
    zz_chain_hamiltonian,
    heisenberg_chain_hamiltonian,
):
    _identity.__module__ = "flagquantum.algorithms.core"
del _identity

_public_namespace = _import_public_namespace("flagquantum.algorithms.core")
for _name in (
    "transverse_field_ising",
    "zz_chain_hamiltonian",
    "heisenberg_chain_hamiltonian",
):
    setattr(_public_namespace, _name, globals()[_name])
del _public_namespace, _name
