"""Construct executable Pauli operations controlled by a bit pattern."""

import re

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
