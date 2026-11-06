# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlScaleModAdd.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py
# Changes: native IR, explicit big-endian roles and a reused clean work pool.

"""Controlled modular scale-add through the original ordered bit traversal."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .modular_addition import _validate_modular_layout
from .modular_constant_addition import (
    controlled_modular_add_constant_adjoint_ir,
    controlled_modular_add_constant_ir,
    modular_constant_requirements,
)


def modular_scaled_requirements(width: int) -> tuple[int, int]:
    """Reserve one outer And target plus the shared constant-add work pool."""
    quantum, classical = modular_constant_requirements(width)
    return quantum + 1, classical


def controlled_modular_scaled_add_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    scale: int,
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Preserve control/x and update y=(y+k*x)%p on the positive control branch.

    Big-endian x may contain any n-bit unsigned value; y lies in [0,p) in both
    coherent control branches. Concrete k may be negative. Do not omit zero-k
    stages: source gate order and resource accounting include every bit.
    Caller-owned work starts/ends zero. Classical slots hold latest outcomes.
    """
    return _controlled_modular_scaled_ir(
        control, left, right, scale, modulus, workspace, classical_bits, adjoint=False
    )


def controlled_modular_scaled_add_adjoint_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    scale: int,
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Subtract k*x modulo p using the exact reverse source traversal.

    Positive coherent control; big-endian x is any n-bit value, y lies in
    [0,p). Visit low to high bits, retain each original normalized constant,
    and use CModAddK's literal adjoint between And and measured erasure.
    Quantum work starts/ends zero; classical slots hold latest outcomes.
    """
    return _controlled_modular_scaled_ir(
        control, left, right, scale, modulus, workspace, classical_bits, adjoint=True
    )


def _controlled_modular_scaled_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    scale: int,
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    requirements = modular_scaled_requirements(len(left))
    _validate_modular_layout(
        left,
        right,
        modulus,
        workspace,
        classical_bits,
        requirements,
        extra_qubits=(control,),
    )
    if type(scale) is not int:
        raise ValueError("Modular scale must be a concrete integer")
    apply, inner = workspace[0], workspace[1:]
    gates: list[Instruction] = []
    indices = range(len(left) - 1, -1, -1) if adjoint else range(len(left))
    emit = (
        controlled_modular_add_constant_adjoint_ir
        if adjoint
        else controlled_modular_add_constant_ir
    )
    for i in indices:
        wire = left[i]
        value = (scale * (1 << (len(left) - 1 - i))) % modulus
        gates.extend(and_compute_ir(control, wire, apply))
        gates.extend(emit(apply, right, value, modulus, inner, classical_bits))
        gates.extend(and_uncompute_ir(control, wire, apply, classical_bits[0]))
    return tuple(gates)
