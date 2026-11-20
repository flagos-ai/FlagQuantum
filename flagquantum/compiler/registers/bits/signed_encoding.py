# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Reference signature and cost model: Qualtran v0.7.0 SignedIntegerToTwosComplement.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/ones_complement_to_twos_complement.py

"""In-place reversible signed encodings for big-endian words."""

from typing import Literal

from ....core.ir import Instruction
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires, controlled_xor_constant_ir


def signed_encoding_workspace_size(width: int) -> int:
    """Return clean carry qubits for a sign and at least one payload bit."""
    if type(width) is not int or width < 2:
        raise ValueError("Signed conversion requires at least two bits")
    return width - 2


def ones_complement_to_twos_ir(
    word: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    measurement_bit: int = 0,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Convert in place, preserving the sign and clearing borrowed carries.

    The all-ones encoding of negative zero maps to the minimum two's
    complement word. The adjoint reverses that full-word permutation.
    """
    return _shift_negative_payload_ir(word, workspace, measurement_bit, adjoint)


def sign_magnitude_to_twos_ir(
    word: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    measurement_bit: int = 0,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Convert sign-magnitude via the sign-controlled complement and increment.

    Negative zero maps to the minimum two's complement word. The adjoint
    subtracts the sign before restoring the magnitude.
    """
    shift = _shift_negative_payload_ir(word, workspace, measurement_bit, adjoint)
    complement = controlled_xor_constant_ir(
        word[0], word[1:], (1 << (len(word) - 1)) - 1
    )
    return (*shift, *complement) if adjoint else (*complement, *shift)


def _shift_negative_payload_ir(
    word: tuple[int, ...],
    workspace: tuple[int, ...],
    measurement_bit: int,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    required = signed_encoding_workspace_size(len(word))
    if len(workspace) != required:
        raise ValueError(f"Signed conversion requires {required} clean carry qubits")
    _validate_wires((*word, *workspace))
    if type(measurement_bit) is not int or measurement_bit < 0:
        raise ValueError("Signed conversion requires a nonnegative measurement slot")

    values = (1, 0 if adjoint else 1)
    operations: list[Instruction] = []
    for kind, wires in _signed_encoding_components(word, workspace):
        if kind == "cx":
            operations.append(Instruction("cx", wires))
        elif kind == "and":
            x, y, target = wires
            operations.extend(and_compute_ir(x, y, target, values=values))
        else:
            x, y, target = wires
            operations.extend(
                and_uncompute_ir(x, y, target, measurement_bit, values=values)
            )
    return tuple(operations)


def _signed_encoding_components(
    word: tuple[int, ...], workspace: tuple[int, ...]
) -> tuple[tuple[Literal["cx", "and", "erase"], tuple[int, ...]], ...]:
    """Share the existing signed increment order between flat and typed paths."""
    sign = word[0]
    payload = word[:0:-1]
    controls = (sign, *workspace)
    steps: list[tuple[Literal["cx", "and", "erase"], tuple[int, ...]]] = [
        ("and", (controls[i], payload[i], carry)) for i, carry in enumerate(workspace)
    ]
    for i in reversed(range(len(workspace))):
        carry = workspace[i]
        steps.append(("cx", (carry, payload[i + 1])))
        steps.append(("erase", (controls[i], payload[i], carry)))
    steps.append(("cx", (sign, payload[0])))
    return tuple(steps)
