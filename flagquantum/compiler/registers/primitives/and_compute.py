# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, And.decompose_from_registers.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/and_bloq.py
# Changes: explicit native IR and caller-owned quantum/classical slots.

"""Shared four-T logical And and measurement-assisted erasure."""

from ....core.ir import Instruction


def _polarities(
    x: int, y: int, target: int, values: tuple[int, int]
) -> tuple[Instruction, ...]:
    if (
        any(type(w) is not int or w < 0 for w in (x, y, target))
        or len({x, y, target}) != 3
    ):
        raise ValueError("And requires three distinct nonnegative quantum wires")
    if len(values) != 2 or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError("And control values must be two concrete zero/one integers")
    return tuple(
        Instruction("x", (w,)) for w, v in zip((x, y), values, strict=True) if v == 0
    )


def and_compute_ir(
    x: int, y: int, target: int, *, values: tuple[int, int] = (1, 1)
) -> tuple[Instruction, ...]:
    """Compute a logical And into a clean target using the literal four-T network."""
    flips = _polarities(x, y, target, values)
    return (
        *flips,
        Instruction("h", (target,)),
        Instruction("t", (target,)),
        Instruction("cx", (x, target)),
        Instruction("cx", (y, target)),
        Instruction("cx", (target, x)),
        Instruction("cx", (target, y)),
        Instruction("tdg", (x,)),
        Instruction("tdg", (y,)),
        Instruction("t", (target,)),
        Instruction("cx", (target, x)),
        Instruction("cx", (target, y)),
        Instruction("h", (target,)),
        Instruction("s", (target,)),
        *flips,
    )


def and_uncompute_ir(
    x: int, y: int, target: int, bit: int, *, values: tuple[int, int] = (1, 1)
) -> tuple[Instruction, ...]:
    """Erase a target storing the specified And, preserving control amplitudes."""
    flips = _polarities(x, y, target, values)
    if type(bit) is not int or bit < 0:
        raise ValueError("And erasure requires a nonnegative integer classical slot")
    return (
        *flips,
        Instruction("h", (target,)),
        Instruction(
            "measure", (target,), metadata={"is_dynamic": True, "classical_bit": bit}
        ),
        Instruction("cz", (x, y), metadata={"conditions": ((bit, 1),)}),
        Instruction("reset", (target,), metadata={"is_dynamic": True}),
        *flips,
    )
