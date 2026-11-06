# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CModAdd.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py
# Changes: explicit reusable work, shared components and native IR.

"""Controlled modular addition through the original conditional sign correction."""

from ...core.ir import Instruction
from .arithmetic import add_constant_ir
from .controlled_addition import controlled_add_constant_ir, controlled_add_ir
from .controlled_comparison import controlled_greater_ir
from .modular_addition import _validate_modular_layout


def controlled_modular_requirements(width: int) -> tuple[int, int]:
    """Return clean quantum work and reusable classical slots for both directions."""
    if type(width) is not int or width < 1:
        raise ValueError(
            "Controlled modular arithmetic requires a positive concrete width"
        )
    return max(2 * width + 2, width + 6), max(2 * width, width + 2)


def _controlled_modular_workspace(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    control_value: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Bind reused roles with a stable classical slot per measured physical qubit."""
    n = len(left)
    requirements = controlled_modular_requirements(n)
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Controlled modular control value must be zero or one")
    _validate_modular_layout(
        left,
        right,
        modulus,
        workspace,
        classical_bits,
        requirements,
        extra_wires=(control,),
    )
    sign, pool = workspace[0], workspace[1:]
    constant, carry = pool[: n + 1], pool[n + 1 : 2 * n + 1]
    # CAdd measures loaded[0:n] then its carry bank, with distinct slots.
    # Reuse that assignment for the comparator's And outputs and final flag.
    measured = (*constant[:n], *carry, constant[-1])[: n + 2]
    spare = tuple(w for w in pool if w not in measured)[:3]
    compare = (*spare[:2], *reversed(measured[:-1]), spare[2], measured[-1])
    return sign, constant, carry, compare


def controlled_modular_add_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Preserve control/x and add x to y modulo p when control matches.

    Equal-width big-endian inputs satisfy x in [0,p], y in [0,p) in both
    coherent control branches. Work starts/ends zero. Keep the unconditional
    source correction stages, including when the selected control is inactive.
    """
    n = len(left)
    sign, constant, carry, compare = _controlled_modular_workspace(
        control, left, right, modulus, workspace, classical_bits, control_value
    )
    load_bits, carry_bits = classical_bits[:n], classical_bits[n : 2 * n]
    return (
        *controlled_add_ir(
            control,
            left,
            (sign, *right),
            constant[:n],
            carry,
            load_bits,
            carry_bits,
            control_value=control_value,
        ),
        *add_constant_ir((sign, *right), -modulus, constant, carry, carry_bits),
        *controlled_add_constant_ir(
            sign, right, modulus, constant[:n], carry[: n - 1], carry_bits[: n - 1]
        ),
        *controlled_greater_ir(
            control,
            left,
            right,
            sign,
            compare,
            classical_bits[: n + 2],
            control_value=control_value,
        ),
        Instruction("x", (sign,)),
    )
