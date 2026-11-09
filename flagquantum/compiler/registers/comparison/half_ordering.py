# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _HalfComparisonBase and LinearDepthHalf*.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: explicit logical outputs and native IR; reuse shared half-comparison.

"""Four source half-comparison relations through input exchange and flag flips."""

from typing import Literal

from ....core.ir import Instruction
from ..bits.bitwise import bitwise_not_ir
from .half_comparison import half_greater_ir


def _reverse_operands(relation: str) -> bool:
    if relation not in ("gt", "ge", "lt", "le"):
        raise ValueError("Half comparison relation must be gt, ge, lt or le")
    return relation in ("ge", "lt")


def half_compare_outputs(
    left: tuple[int, ...],
    right: tuple[int, ...],
    difference: tuple[int, ...],
    target: int,
    *,
    relation: Literal["gt", "ge", "lt", "le"] = "gt",
    adjoint: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Bind logical ports for the same validated layout passed to half_compare_ir.

    Preserve the pinned executable port aliases: ge/lt adjoints return a on
    the original right qubits and b on the left, although the physical state
    is restored. This source decomposition differs from its classical model;
    callers must retain these bindings rather than silently assume identity.
    """
    reverse = _reverse_operands(relation)
    if type(adjoint) is not bool:
        raise ValueError("Half comparison adjoint must be boolean")
    if adjoint:
        return {"a": right, "b": left} if reverse else {"a": left, "b": right}
    return {"a": left, "b": right, "c": difference, "target": (target,)}


def half_compare_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    difference: tuple[int, ...],
    target: int,
    signs: tuple[int, ...],
    *,
    relation: Literal["gt", "ge", "lt", "le"] = "gt",
    signed: bool = False,
    adjoint: bool = False,
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Compute a comparison and retained difference, or erase the correct pair.

    gt/le retain b-a, while ge/lt retain a-b, modulo 2**(n+1).
    Output/initialization and measured-adjoint contracts are half_greater_ir's.
    Physical input qubits and the original pre/post flip order are preserved.
    Logical ports must be read through half_compare_outputs, including the
    pinned ge/lt adjoint output-alias exchange.
    """
    reverse = _reverse_operands(relation)
    a, b = (right, left) if reverse else (left, right)
    inner = half_greater_ir(
        a,
        b,
        difference,
        target,
        signs,
        signed=signed,
        adjoint=adjoint,
        classical_bits=classical_bits,
    )
    flip = bitwise_not_ir((target,)) if relation in ("ge", "le") else ()
    return (*flip, *inner) if adjoint else (*inner, *flip)
