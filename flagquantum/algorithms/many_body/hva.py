"""Heisenberg Hamiltonian variational ansatz circuits."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module as _import_public_namespace

import torch

from flagquantum.circuit import Circuit


def heisenberg_hva_parameter_count(
    n_qubits: int, depth: int, *, parameterization: str = "bond_resolved_phase"
) -> int:
    if n_qubits < 2 or depth < 1:
        raise ValueError("n_qubits >= 2 and depth >= 1 are required")
    if parameterization == "shared_parity":
        return 6 * depth
    if parameterization == "bond_resolved":
        return 3 * (n_qubits - 1) * depth
    if parameterization == "bond_resolved_phase":
        return (3 * (n_qubits - 1) + n_qubits) * depth
    raise ValueError("unknown Heisenberg HVA parameterization")


def heisenberg_hva(
    n_qubits: int,
    depth: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    parameterization: str = "bond_resolved_phase",
    initial_state: str = "dimer_singlet",
) -> Circuit:
    """Build an open-chain HVA with a physics-informed initial state."""

    values = torch.as_tensor(parameters)
    expected = heisenberg_hva_parameter_count(
        n_qubits, depth, parameterization=parameterization
    )
    if values.numel() != expected:
        raise ValueError(f"expected {expected} HVA parameters, got {values.numel()}")
    values = values.reshape(-1)
    circuit = Circuit(n_qubits, device=values.device)
    if initial_state == "neel":
        for qubit in range(1, n_qubits, 2):
            circuit.gate("x", qubit)
    elif initial_state == "dimer_singlet":
        for left in range(0, n_qubits - 1, 2):
            circuit.gate("x", left + 1)
            circuit.gate("h", left)
            circuit.gate("cx", (left, left + 1))
            circuit.gate("z", left)
    else:
        raise ValueError("initial_state must be 'neel' or 'dimer_singlet'")

    for layer in range(depth):
        if parameterization == "shared_parity":
            offset = 6 * layer
            for axis, gate_name in enumerate(("rxx", "ryy", "rzz")):
                for parity in (0, 1):
                    theta = values[offset + 2 * axis + parity]
                    for left in range(parity, n_qubits - 1, 2):
                        circuit.gate(gate_name, (left, left + 1), theta=theta)
            continue
        per_layer = 3 * (n_qubits - 1) + (
            n_qubits if parameterization == "bond_resolved_phase" else 0
        )
        offset = per_layer * layer
        for axis, gate_name in enumerate(("rxx", "ryy", "rzz")):
            for left in range(n_qubits - 1):
                circuit.gate(
                    gate_name,
                    (left, left + 1),
                    theta=values[offset + axis * (n_qubits - 1) + left],
                )
        if parameterization == "bond_resolved_phase":
            phase_offset = offset + 3 * (n_qubits - 1)
            for qubit in range(n_qubits):
                circuit.gate("rz", qubit, theta=values[phase_offset + qubit])
    return circuit


__all__ = ("heisenberg_hva_parameter_count", "heisenberg_hva")

# Preserve the public algorithm and serialized result identities.
for _identity_export in (
    heisenberg_hva_parameter_count,
    heisenberg_hva,
):
    _identity_export.__module__ = "flagquantum.algorithms.core"
del _identity_export

# Bind the historical public namespace for class annotation introspection.

_public_namespace = _import_public_namespace("flagquantum.algorithms.core")
for _public_name in ("heisenberg_hva", "heisenberg_hva_parameter_count"):
    setattr(_public_namespace, _public_name, globals()[_public_name])
del _public_namespace, _public_name
