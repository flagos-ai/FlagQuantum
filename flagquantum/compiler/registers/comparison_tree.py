# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from pinned Qualtran v0.7.0 BiQubitsMixer and SingleQubitCompare.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: native IR and explicit logical outputs/measurement slots.

"""Compute/erase components of the upstream comparison tree."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir


def _validate(
    qubits: tuple[int, ...], bits: tuple[int, ...], count: int, adjoint: bool
) -> None:
    if type(adjoint) is not bool:
        raise ValueError("Comparison adjoint flag must be boolean")
    if any(type(w) is not int or w < 0 for w in qubits) or len(set(qubits)) != len(
        qubits
    ):
        raise ValueError("Comparison components require distinct nonnegative wires")
    expected = count if adjoint else 0
    if (
        len(bits) != expected
        or len(set(bits)) != expected
        or any(type(b) is not int or b < 0 for b in bits)
    ):
        raise ValueError(
            f"Comparison component requires {expected} distinct nonnegative classical slots"
        )


def single_qubit_compare_ir(
    a: int,
    b: int,
    less: int,
    greater: int,
    *,
    adjoint: bool = False,
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Map clean flags to a<b/a>b, storing equality in b; adjoint restores inputs."""
    _validate((a, b, less, greater), classical_bits, 1, adjoint)
    middle = (
        Instruction("cx", (less, greater)),
        Instruction("cx", (b, greater)),
        Instruction("cx", (a, b)),
        Instruction("cx", (a, greater)),
        Instruction("x", (b,)),
    )
    if adjoint:
        return (
            *reversed(middle),
            *and_uncompute_ir(a, b, less, classical_bits[0], values=(0, 1)),
        )
    return (*and_compute_ir(a, b, less, values=(0, 1)), *middle)


def _swap(
    control: int, a: int, b: int, aux: int, bit: int | None
) -> tuple[Instruction, ...]:
    if bit is None:
        return (
            Instruction("cx", (a, b)),
            *and_compute_ir(control, b, aux),
            Instruction("cx", (aux, a)),
            Instruction("cx", (a, b)),
        )
    return (
        Instruction("cx", (a, b)),
        Instruction("cx", (aux, a)),
        *and_uncompute_ir(control, b, aux, bit),
        Instruction("cx", (a, b)),
    )


def bi_qubits_mixer_ir(
    x: tuple[int, ...],
    y: tuple[int, ...],
    work: tuple[int, ...],
    *,
    adjoint: bool = False,
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Preserve comparison sign in low bits; work is logical output until erased.

    Each input has two bits, with three initially zero work qubits. The exact
    source adjoint requires the forward output (including all three work flags)
    and restores the inputs and work, measuring only the last two flags.
    """
    if len(x) != 2 or len(y) != 2 or len(work) != 3:
        raise ValueError("Mixer requires two two-bit inputs and three work wires")
    _validate((*x, *y, *work), classical_bits, 2, adjoint)
    xm, xl = x
    ym, yl = y
    p, u, v = work
    initial = (
        Instruction("x", (p,)),
        Instruction("cx", (ym, xm)),
        Instruction("cx", (yl, xl)),
    )
    final = Instruction("cx", (yl, xl))
    if adjoint:
        return (
            final,
            *_swap(xm, ym, yl, v, classical_bits[1]),
            *_swap(xm, xl, p, u, classical_bits[0]),
            *reversed(initial),
        )
    return (*initial, *_swap(xm, xl, p, u, None), *_swap(xm, ym, yl, v, None), final)
