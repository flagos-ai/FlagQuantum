# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CAdd.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/controlled_addition.py
# Changes: native IR and explicit quantum/classical workspace.

"""Controlled register addition and exact adjoints through shared source emitters."""

from ....core.ir import Instruction
from ..primitives.and_compute import and_compute_ir, and_uncompute_ir
from .addition import _constant_addition_ir, add_adjoint_ir, add_ir


def controlled_add_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    loaded: tuple[int, ...],
    carries: tuple[int, ...],
    load_bits: tuple[int, ...],
    carry_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Add a zero-extended word on the selected coherent control branch.

    Supply one clean loaded qubit per left bit and len(right)-1 clean carries,
    each with a distinct classical slot. Quantum work starts/ends zero; the
    slots retain latest measurement outcomes.
    Unequal-width bit patterns use zero extension, not signed extension.
    """
    return _controlled_addition_ir(
        control,
        left,
        right,
        loaded,
        carries,
        load_bits,
        carry_bits,
        control_value=control_value,
        adjoint=False,
    )


def controlled_add_adjoint_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    loaded: tuple[int, ...],
    carries: tuple[int, ...],
    load_bits: tuple[int, ...],
    carry_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Port CAdd.adjoint with original loaded bits and exact shared Add adjoint.

    Supply one clean loaded qubit per left bit and len(right)-1 clean carries,
    each with a distinct classical slot. Quantum work starts/ends zero; the
    slots retain latest measurement outcomes.
    Unequal-width bit patterns use zero extension, not signed extension.
    """
    return _controlled_addition_ir(
        control,
        left,
        right,
        loaded,
        carries,
        load_bits,
        carry_bits,
        control_value=control_value,
        adjoint=True,
    )


def _controlled_addition_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    loaded: tuple[int, ...],
    carries: tuple[int, ...],
    load_bits: tuple[int, ...],
    carry_bits: tuple[int, ...],
    *,
    control_value: int,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    """Share loaded-word validation, compute, exact Add/adjoint and erase stages.

    All registers are big-endian. Supply m clean loaded qubits, n-1 clean carries
    for m<=n, and one distinct classical measurement slot for each work qubit.
    Left and control retain their values; right is updated and all work cleared.
    Equal-width bit patterns also implement signed addition modulo 2**n.
    Unequal-width signed sign extension is not supplied by this entry point.
    """
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Controlled addition control value must be zero or one")
    if len(loaded) != len(left) or not left:
        raise ValueError("Controlled addition requires one loaded wire per input bit")
    wires = (control, *left, *right, *loaded, *carries)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError(
            "Controlled addition requires distinct nonnegative quantum wires"
        )
    bits = (*load_bits, *carry_bits)
    if (
        len(load_bits) != len(loaded)
        or len(set(bits)) != len(bits)
        or any(type(bit) is not int or bit < 0 for bit in bits)
    ):
        raise ValueError(
            "Controlled addition requires distinct nonnegative classical slots"
        )
    emit = add_adjoint_ir if adjoint else add_ir
    addition = emit(loaded, right, carries, carry_bits)
    values = (control_value, 1)
    compute = tuple(
        gate
        for a, work in zip(left, loaded, strict=True)
        for gate in and_compute_ir(control, a, work, values=values)
    )
    erase = tuple(
        gate
        for a, work, bit in reversed(tuple(zip(left, loaded, load_bits, strict=True)))
        for gate in and_uncompute_ir(control, a, work, bit, values=values)
    )
    return (*compute, *addition, *erase)


def controlled_add_constant_ir(
    control: int,
    target: tuple[int, ...],
    value: int,
    constant: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Conditionally add an unsigned constant by controlling only its load.

    Work allocation and normalization match add_constant_ir. The shared Add
    executes on both control branches, as in upstream Always(Add). Control-on-zero
    retains the source's global X conjugation. All quantum work starts/ends zero.
    """
    return _controlled_constant_ir(
        control,
        target,
        value,
        constant,
        carries,
        classical_bits,
        control_value=control_value,
        adjoint=False,
    )


def controlled_add_constant_adjoint_ir(
    control: int,
    target: tuple[int, ...],
    value: int,
    constant: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Port controlled AddK(k).adjoint(), keeping the same controlled load k.

    Subtract k when control matches, preserving control phases and clearing all
    constant/carry work. The source's zero-control outer X pair is retained.
    """
    return _controlled_constant_ir(
        control,
        target,
        value,
        constant,
        carries,
        classical_bits,
        control_value=control_value,
        adjoint=True,
    )


def _controlled_constant_ir(
    control: int,
    target: tuple[int, ...],
    value: int,
    constant: tuple[int, ...],
    carries: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    """One controlled constant body and polarity policy for both directions."""
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError(
            "Controlled constant addition control value must be zero or one"
        )
    if type(control) is not int or control < 0:
        raise ValueError(
            "Controlled constant addition requires a nonnegative control wire"
        )
    body = _constant_addition_ir(
        target,
        value,
        constant,
        carries,
        classical_bits,
        control=control,
        adjoint=adjoint,
    )
    flip = (Instruction("x", (control,)),) if control_value == 0 else ()
    return (*flip, *body, *flip)
