# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CModNeg.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py
# Changes: native IR and shared zero-detection/fanout/controlled AddK components.

"""Controlled modular negation retaining the source apply-And lifetime."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .controlled_addition import controlled_add_constant_ir
from .modular_negation import _negation_layout, modular_negation_requirements
from .multi_control import multi_control_x_ir
from .sign_extension import _fanout_ir


def controlled_modular_negation_requirements(width: int) -> tuple[int, int]:
    """Reserve the source apply target in addition to ordinary modular work."""
    q, c = modular_negation_requirements(width)
    return q + 1, max(1, c)


def controlled_modular_negate_ir(
    control: int,
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Preserve coherent control and negate a valid residue on either polarity.

    Both control branches promise 0<=x<p, with p+1 fitting n bits. Work starts
    and ends zero; retain the source zero checks even on the inactive branch.
    """
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Modular negation control value must be zero or one")
    flag, constant, carry = _negation_layout(
        target, modulus, workspace, classical_bits, control=control
    )
    apply = workspace[1]
    bits = classical_bits[: len(target) - 1]
    zero = multi_control_x_ir(target, (0,) * len(target), flag, carry, bits)
    return (
        Instruction("x", (flag,)),
        *zero,
        *and_compute_ir(control, flag, apply, values=(control_value, 1)),
        *_fanout_ir(apply, target),
        *controlled_add_constant_ir(apply, target, modulus + 1, constant, carry, bits),
        *and_uncompute_ir(
            control, flag, apply, classical_bits[0], values=(control_value, 1)
        ),
        *zero,
        Instruction("x", (flag,)),
    )
