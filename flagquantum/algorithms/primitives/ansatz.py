"""Reusable hardware-efficient circuit constructions."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module as _import_public_namespace

import torch

from flagquantum.circuit import Circuit
from flagquantum.core.runtime_config import get_runtime_config


def hardware_efficient_parameter_count(
    n_qubits: int,
    layers: int,
    rotations: Sequence[str] = ("ry", "rz"),
) -> int:
    return int(n_qubits) * int(layers) * len(tuple(rotations))


def hardware_efficient_ansatz(
    n_qubits: int,
    layers: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    rotations: Sequence[str] = ("ry", "rz"),
    entanglement: str = "linear",
    device: torch.device | str = "cpu",
) -> Circuit:
    """Build a common differentiable layered ansatz."""

    params = torch.as_tensor(parameters, dtype=torch.float32, device=device).reshape(-1)
    expected = hardware_efficient_parameter_count(n_qubits, layers, rotations)
    if params.numel() != expected:
        raise ValueError(f"Expected {expected} parameters, got {params.numel()}.")

    circuit = Circuit(
        n_qubits,
        device=device,
        dtype=getattr(torch, get_runtime_config().complex_dtype),
    )
    cursor = 0
    for _ in range(layers):
        for qubit in range(n_qubits):
            for rotation in rotations:
                getattr(circuit, rotation.lower())(qubit, theta=params[cursor])
                cursor += 1
        if entanglement == "linear":
            for qubit in range(n_qubits - 1):
                circuit.gate("cx", (qubit, qubit + 1))
        elif entanglement == "circular":
            for qubit in range(n_qubits - 1):
                circuit.gate("cx", (qubit, qubit + 1))
            if n_qubits > 2:
                circuit.gate("cx", (n_qubits - 1, 0))
        elif entanglement != "none":
            raise ValueError("entanglement must be 'linear', 'circular', or 'none'.")
    return circuit


__all__ = ("hardware_efficient_parameter_count", "hardware_efficient_ansatz")

# Preserve the public algorithm and serialized result identities.
for _identity_export in (
    hardware_efficient_parameter_count,
    hardware_efficient_ansatz,
):
    _identity_export.__module__ = "flagquantum.algorithms.core"
del _identity_export

# Bind the historical public namespace for class annotation introspection.

_public_namespace = _import_public_namespace("flagquantum.algorithms.core")
for _public_name in ("hardware_efficient_ansatz", "hardware_efficient_parameter_count"):
    setattr(_public_namespace, _public_name, globals()[_public_name])
del _public_namespace, _public_name
