# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/large_bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/large_bloq.py

"""A measured ladder fixture with each polarity on its incoming prefix."""

from flagquantum.compiler.registers.layout.register_bits import split_integer
from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature


def large_fixture_signature(n_select: int) -> Signature:
    return Signature.build(select=n_select, target=1)


def large_fixture_ir(
    select: tuple[int, ...],
    target: int,
    repetitions: int,
    workspace: tuple[int, ...],
    readouts: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Build the source ladder with a controlled X as its atomic action."""
    depth = len(select) - 1
    if depth < 0 or repetitions < 0:
        raise ValueError(
            "The fixture requires at least one select bit and nonnegative repetitions"
        )
    if (
        len(workspace) != (depth if repetitions else 0)
        or len(readouts) != depth * repetitions
    ):
        raise ValueError("Workspace and readout counts must match the ladder")
    wires = (*select, target, *workspace)
    if len(set(wires)) != len(wires) or len(set(readouts)) != len(readouts):
        raise ValueError("The ladder requires distinct wires and readouts")

    gates: list[Instruction] = []
    for index in range(repetitions):
        pattern = split_integer(index % depth, depth) if depth else ()
        steps = tuple(
            (prefix, bit, ancilla, (value, 1))
            for prefix, bit, ancilla, value in zip(
                (select[0], *workspace[:-1])[:depth],
                select[1:],
                workspace,
                pattern,
                strict=True,
            )
        )
        for prefix, bit, ancilla, values in steps:
            gates.extend(and_compute_ir(prefix, bit, ancilla, values=values))
        gates.append(Instruction("cx", (workspace[-1] if depth else select[0], target)))
        for offset, (prefix, bit, ancilla, values) in enumerate(reversed(steps)):
            gates.extend(
                and_uncompute_ir(
                    prefix,
                    bit,
                    ancilla,
                    readouts[index * depth + offset],
                    values=values,
                )
            )
    return tuple(gates)
