# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 t_factory_utils.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/surface_code/t_factory_utils.py

"""Noise channels used by the magic-state distillation models."""

from math import cos, pi, sin, sqrt

import torch

from ...core.ir import Instruction
from ...noise.channels import (
    KrausChannel,
    _pauli_basis,
    bit_flip_channel,
    phase_flip_channel,
)

_DTYPE = torch.complex128


def noisy_pauli_rotation(
    paulis: str, p1: float, p2: float, p3: float, wires: tuple[int, ...]
) -> Instruction:
    """Apply the four rotations in source order on big-endian Pauli operands."""
    if not paulis or len(paulis) != len(wires) or set(paulis) - set("IXYZ"):
        raise ValueError("Expected a nonempty Pauli word matching the wires")
    probabilities = (1 - p1 - p2 - p3, p1, p2, p3)
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Rotation probabilities must define a valid mixture")
    basis = dict(zip("IXYZ", _pauli_basis(dtype=_DTYPE, device="cpu"), strict=True))
    pauli = basis[paulis[0]]
    for letter in paulis[1:]:
        pauli = torch.kron(pauli, basis[letter])
    identity = torch.eye(pauli.shape[0], dtype=_DTYPE, device="cpu")
    kraus = tuple(
        sqrt(probability) * (cos(angle) * identity + 1j * sin(angle) * pauli)
        for probability, angle in zip(
            probabilities, (pi / 8, 5 * pi / 8, -pi / 8, 3 * pi / 8), strict=True
        )
    )
    channel = KrausChannel("noisy_pauli_rotation", kraus)
    return Instruction(
        channel.name, wires, matrix=channel.kraus, metadata={"is_channel": True}
    )


def storage_error(
    kind: str, probabilities: tuple[float, ...], wires: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Apply each independent X or Z storage error to its declared wire."""
    if kind not in ("X", "Z"):
        raise ValueError("Storage error must be X or Z")
    channel_factory = bit_flip_channel if kind == "X" else phase_flip_channel
    instructions = []
    for probability, wire in zip(probabilities, wires, strict=True):
        channel = channel_factory(
            torch.tensor(probability, dtype=torch.float64), dtype=_DTYPE, device="cpu"
        )
        instructions.append(
            Instruction(
                channel.name,
                (wire,),
                matrix=channel.kraus,
                metadata={"is_channel": True},
            )
        )
    return tuple(instructions)
