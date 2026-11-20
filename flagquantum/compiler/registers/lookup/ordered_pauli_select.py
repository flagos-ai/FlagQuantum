# Copyright 2024 Google LLC
# Copyright 2018 The Cirq Developers
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TestPauliSelectOracle.decompose_from_registers
# and the Cirq controlled-Pauli dispatch reached by that source.

"""Ordered little-endian Pauli lookup without clean work or measurement slots."""

from cmath import phase
from math import isclose, pi

from ....core.ir import Instruction
from ...optimization.controlled_unitary_synthesis import _phase
from ..bits.bitwise import _validate_wires
from ..primitives.controlled_hadamard import (
    _ccz_phase,
    _controlled_matrix,
    _diagonal_phase,
)


def _pauli(
    name: str, controls: tuple[int, ...], values: tuple[int, ...], target: int
) -> tuple[Instruction, ...]:
    if name == "I":
        return ()
    if values == (1,):
        return (Instruction("c" + name.lower(), (*controls, target)),)
    if values == (1, 1):
        if name == "Z":
            return _ccz_phase((*controls, target), 1)
        middle = Instruction("ccx", (*controls, target))
        if name == "Y":
            return (Instruction("sdg", (target,)), middle, Instruction("s", (target,)))
        return (middle,)
    matrix = {
        "X": [[0j, 1 + 0j], [1 + 0j, 0j]],
        "Y": [[0j, -1j], [1j, 0j]],
        "Z": [[1 + 0j, 0j], [0j, -1 + 0j]],
    }[name]
    return _controlled_matrix(matrix, controls, values, target)


def _coefficient(
    coefficient: complex, controls: tuple[int, ...], values: tuple[int, ...]
) -> tuple[Instruction, ...]:
    if coefficient == 1:
        return ()
    # Source Pauli inversion canonicalizes the imaginary signed zero at -1.
    coefficient = complex(coefficient.real, 0) if coefficient.imag == 0 else coefficient
    angle = phase(coefficient)
    if not values[-1]:
        return _diagonal_phase(controls, values, angle)
    if values == (1,):
        return (_phase(controls[0], angle / pi),)
    if values == (1, 1):
        return (Instruction("cphase", controls, {"theta": angle}),)
    if values == (1, 1, 1):
        return _ccz_phase(controls, angle / pi)
    return _controlled_matrix(
        [[1 + 0j, 0j], [0j, coefficient]], controls[:-1], values[:-1], controls[-1]
    )


def ordered_pauli_select_ir(
    words: tuple[str, ...],
    selection: tuple[int, ...],
    target: tuple[int, ...],
    *,
    control: tuple[int, ...] = (),
    control_value: int = 1,
    coefficients: tuple[complex, ...] | None = None,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Emit each controlled Pauli in source address and target order.

    The first selection wire is the least-significant address bit. An optional
    outer one-bit control precedes the selection controls. Each table row has
    one I/X/Y/Z per target and an optional unit-modulus scalar. Omitted addresses
    act as identity. Workspace and classical-slot requirements are zero. Adjoint reverses rows and conjugates scalars.
    Local 2x2 matrix leaves retain the existing recursive synthesis boundary.
    """
    _validate_wires((*control, *selection, *target))
    if not selection or not target or len(control) > 1:
        raise ValueError(
            "Ordered Pauli lookup requires nonempty selection/target and at most one control"
        )
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Ordered Pauli lookup requires a binary control value")
    if type(adjoint) is not bool:
        raise ValueError("Ordered Pauli lookup direction must be Boolean")
    if len(words) > 1 << len(selection) or any(
        len(word) != len(target) or any(letter not in "IXYZ" for letter in word)
        for word in words
    ):
        raise ValueError(
            "Ordered Pauli table must fit the address range with valid target words"
        )
    scalars = (1 + 0j,) * len(words) if coefficients is None else coefficients
    if len(scalars) != len(words) or any(
        not isclose(abs(value), 1, rel_tol=0, abs_tol=1e-12) for value in scalars
    ):
        raise ValueError("Ordered Pauli coefficients must have unit modulus")
    controls = (*control, *selection)
    gates: list[Instruction] = []
    indices = range(len(words) - 1, -1, -1) if adjoint else range(len(words))
    for address in indices:
        values = ((control_value,) if control else ()) + tuple(
            (address >> bit) & 1 for bit in range(len(selection))
        )
        for letter, wire in zip(words[address], target, strict=True):
            gates.extend(_pauli(letter, controls, values, wire))
        scalar = scalars[address].conjugate() if adjoint else scalars[address]
        gates.extend(_coefficient(scalar, controls, values))
    return tuple(gates)
