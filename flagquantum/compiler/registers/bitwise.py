# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Xor, XorK and BitwiseNot.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py
# OnEach source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/on_each.py
# Fanout source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_target_cnot.py
# Changes: explicit big-endian qubit tuples, concrete unsigned constants and
# one positive controlled XorK, all emitting native IR.

"""Shared register bit permutations without hidden work or dtype conversion."""

from ...core.classical_control import is_classical_control_instruction
from ...core.ir import Instruction


def _validate_wires(qubits: tuple[int, ...]) -> None:
    if not qubits:
        raise ValueError("Bitwise operation requires nonempty registers")
    if any(type(w) is not int or w < 0 for w in qubits) or len(set(qubits)) != len(
        qubits
    ):
        raise ValueError("Bitwise operation requires distinct nonnegative wires")


def xor_ir(source: tuple[int, ...], target: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Preserve source and XOR it into any equal-width target, MSB first."""
    if not source or len(source) != len(target):
        raise ValueError("XOR requires equal nonempty register widths")
    _validate_wires((*source, *target))
    return tuple(Instruction("cx", (a, b)) for a, b in zip(source, target, strict=True))


def _validate_unsigned_constant(width: int, value: int) -> None:
    if type(value) is not int or not 0 <= value < 1 << width:
        raise ValueError("XOR constant must fit the unsigned register")


def xor_constant_ir(target: tuple[int, ...], value: int) -> tuple[Instruction, ...]:
    """XOR a concrete unsigned bit pattern into a big-endian register."""
    _validate_wires(target)
    _validate_unsigned_constant(len(target), value)
    return tuple(
        Instruction("x", (wire,))
        for i, wire in enumerate(target)
        if (value >> (len(target) - i - 1)) & 1
    )


def on_each_ir(gate: Instruction, target: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Bind a single-wire gate on each wire in register order.

    An empty register emits no gates. The bound leaf must carry no classical
    operands, measurement, reset, or channel behavior.
    """
    if (
        len(gate.wires) != 1
        or gate.name in {"measure", "reset"}
        or gate.metadata.get("is_channel")
        or is_classical_control_instruction(gate)
    ):
        raise ValueError("OnEach requires one unconditioned single-qubit gate")
    if target:
        _validate_wires(target)
    return tuple(
        Instruction(
            gate.name,
            (wire,),
            params=gate.params,
            matrix=gate.matrix,
            metadata=gate.metadata,
        )
        for wire in target
    )


def bitwise_not_ir(target: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Flip every bit, preserving the source OnEach order."""
    if not target:
        raise ValueError("Bitwise operation requires nonempty registers")
    return on_each_ir(Instruction("x", (0,)), target)


def controlled_xor_constant_ir(
    control: int, target: tuple[int, ...], value: int
) -> tuple[Instruction, ...]:
    """Apply a concrete XorK under one positive control, in original bit order.

    The control may be coherent. No conjunction work, measurement or T gates
    are introduced. Other control patterns are outside this concrete entry.
    """
    flips = xor_constant_ir(target, value)
    if type(control) is not int or control < 0 or control in target:
        raise ValueError("Controlled XOR requires a distinct nonnegative control")
    return tuple(Instruction("cx", (control, gate.wires[0])) for gate in flips)


def fanout_ir(control: int, targets: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Retain MultiTargetCNOT's depth-layer ordering, including dirty-target fanout."""
    depth = len(targets).bit_length()
    layers = [
        tuple(
            Instruction("cx", (c, t))
            for c, t in zip(
                targets[: 2**i],
                targets[2**i : min(len(targets), 2 ** (i + 1))],
                strict=False,
            )
        )
        for i in range(depth)
    ]
    return (
        *(gate for layer in reversed(layers) for gate in layer),
        Instruction("cx", (control, targets[0])),
        *(gate for layer in layers for gate in layer),
    )
