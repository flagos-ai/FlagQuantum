"""Construct executable Pauli operations controlled by a bit pattern."""

import re
from collections.abc import Mapping
from typing import Any

import torch


def decode_controlled_pauli(name: str) -> tuple[str, str] | None:
    """Read the control pattern and Pauli from the canonical operation name."""
    match = re.fullmatch(r"c([01]+)([xz])", name)
    return (match[1], match[2].upper()) if match else None


def controlled_pauli(bits: str, pauli: str) -> tuple[str, torch.Tensor]:
    """Return a gate name and matrix for Circuit.gate or dynamic feedback.

    The bits follow the supplied control-qubit order. The target qubit is last.
    """
    name = f"c{bits}{pauli.lower()}"
    if decode_controlled_pauli(name) is None:
        raise ValueError("Expected a nonempty binary pattern and Pauli X or Z.")
    matrix = torch.eye(2 ** (len(bits) + 1), dtype=torch.complex64)
    start = 2 * int(bits, 2)
    block = [[0, 1], [1, 0]] if pauli.upper() == "X" else [[1, 0], [0, -1]]
    matrix[start : start + 2, start : start + 2] = torch.tensor(block)
    return name, matrix


def lower_controls(
    name: str,
    qubits: tuple[int, ...],
    controls: Mapping[int, int],
    params: Mapping[str, Any] | None,
    matrix: Any | None,
) -> tuple[str, tuple[int, ...], Any, dict[str, Any]]:
    """Lower a structured control to the existing matrix execution path."""
    from flagquantum.errors import ValidationError

    if not controls:
        return name, qubits, matrix, {}
    if name not in {"x", "z"} or len(qubits) != 1 or params or matrix is not None:
        raise ValidationError(
            "Quantum controls currently require a single X or Z target."
        )
    pairs = tuple(controls.items())
    if any(
        type(qubit) is not int or qubit < 0 or type(bit) is not int or bit not in (0, 1)
        for qubit, bit in pairs
    ):
        raise ValidationError(
            "Quantum controls require integer qubits and binary values."
        )
    if qubits[0] in controls:
        raise ValidationError("A target qubit cannot also be a control.")
    bits = "".join(str(bit) for _, bit in pairs)
    _, unitary = controlled_pauli(bits, name)
    return (
        "controlled_pauli",
        tuple(controls) + qubits,
        unitary,
        {"quantum_controls": pairs, "target_gate": name},
    )
