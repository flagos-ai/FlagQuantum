# Portions adapted from Cirq 1.7.0, two_qubit_to_cz.py and
# analytical_decompositions/single_qubit_decompositions.py.
# Copyright 2018 The Cirq Developers
# Licensed under the Apache License, Version 2.0. See the repository LICENSE or
# https://www.apache.org/licenses/LICENSE-2.0.
# Modified for FlagQuantum: native IR, RX/RZ-only Euler emission, existing matrix
# readers, exact zero tests, bounded generated blocks, and independent acceptance.
# Source revision and evidence: docs/development/COMPILER_CZ_CIRQ_REUSE.md.

"""Compact numeric local runs for a full-CZ/arbitrary-RX/RZ target.

This is proposal construction inside the verified CZ block stage. Its output is
valid up to global phase and must pass that stage's independent 4x4 check. It is
not suitable for targets that publish only fixed-angle RX pulses.
"""

from __future__ import annotations

import math

from ...core.ir import Instruction
from .one_qubit_optimization import Matrix, _instruction_matrix, _matmul
from .one_qubit_synthesis import _principal, _zyz_angles


def _rx_rz_leaves(matrix: Matrix, wire: int) -> tuple[Instruction, ...]:
    # Cirq's Z-X/Y-Z construction, restricted to the admitted RX/RZ vocabulary.
    # Reuse FQ's authoritative matrix-to-Euler reader instead of another solver.
    theta, phi, lam = _zyz_angles(matrix)
    before = _principal(lam - math.pi / 2)
    after = _principal(phi + math.pi / 2)
    rotations: tuple[tuple[str, float], ...]
    if theta == 0.0:
        rotations = (("rz", _principal(phi + lam)),)
    elif theta == math.pi:
        rotations = (("rz", _principal(before - after)), ("rx", theta))
    else:
        rotations = (("rz", before), ("rx", theta), ("rz", after))
    # No tolerance-based gate deletion: the enclosing numerical verifier owns
    # approximation acceptance, and a supplied nonzero rotation is not identity.
    return tuple(
        Instruction(opcode, (wire,), {"theta": angle})
        for opcode, angle in rotations
        if angle != 0.0
    )


def compact_native_cz_locals(
    instructions: tuple[Instruction, ...], pair: tuple[int, int]
) -> tuple[Instruction, ...]:
    """Re-synthesize generated local runs without increasing their gate count."""
    if any(
        item.metadata
        or item.matrix is not None
        or not set(item.wires) <= set(pair)
        or (item.name == "cz" and len(item.wires) != 2)
        or (
            item.name != "cz"
            and (item.name not in {"rx", "rz"} or len(item.wires) != 1)
        )
        for item in instructions
    ):
        return instructions
    pending: dict[int, list[Instruction]] = {wire: [] for wire in pair}
    products: dict[int, Matrix] = {wire: [[1 + 0j, 0j], [0j, 1 + 0j]] for wire in pair}
    output: list[Instruction] = []

    def flush(wire: int) -> None:
        run = pending[wire]
        if run:
            candidate = _rx_rz_leaves(products[wire], wire)
            output.extend(candidate if len(candidate) < len(run) else run)
            pending[wire] = []
            products[wire] = [[1 + 0j, 0j], [0j, 1 + 0j]]

    # Adapt Cirq's pending-per-qubit traversal. A CZ closes both local runs;
    # no phase is moved through a two-qubit operation or absorbed into a layout.
    for instruction in instructions:
        if instruction.name == "cz":
            for wire in pair:
                flush(wire)
            output.append(instruction)
        else:
            matrix = _instruction_matrix(instruction)
            if matrix is None:
                return instructions
            wire = instruction.wires[0]
            products[wire] = _matmul(matrix, products[wire])
            pending[wire].append(instruction)
    for wire in pair:
        flush(wire)
    return tuple(output) if len(output) < len(instructions) else instructions
