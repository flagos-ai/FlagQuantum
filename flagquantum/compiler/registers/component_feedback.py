# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Classical control semantics adapted from Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/classically_controlled.py

"""Owned classical controls composed with existing conditional quantum gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import count
from math import prod

from ...core.classical_control import is_classical_control_instruction
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import ConditionRef, FunctionRecord, Operation, ValueRef
from ...core.ir.program.register_types import classical_bit_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from .component_builder import BindingCollection, ComponentProgramBuilder
from .component_gate_function import bit_gate_function


def classical_control_signature(
    child: Signature,
    controls: tuple[tuple[RegisterType, tuple[SymbolicInt, ...]], ...],
) -> Signature:
    """Prepend ordered classical data ports with the locked collision rule."""
    for dtype, _ in controls:
        if dtype != classical_bit_type() and (
            dtype.quantum_bits != 0 or ("storage", "classical") not in dtype.parameters
        ):
            raise ValueError("Feedback control requires a classical data type")
    occupied = {reg.name for reg in child}
    names: tuple[str, ...]
    if len(controls) == 1 and "ctrl" not in occupied:
        names = ("ctrl",)
    else:
        candidates = (
            f"ctrl{i}"
            for i in count(2 if "ctrl" in occupied else 1)
            if f"ctrl{i}" not in occupied
        )
        names = tuple(next(candidates) for _ in controls)
    return Signature(
        (
            *(
                Register(name, dtype, shape)
                for name, (dtype, shape) in zip(names, controls, strict=True)
            ),
            *child,
        )
    )


def call_classical_feedback(
    builder: ComponentProgramBuilder,
    function: FunctionRecord,
    control_bindings: tuple[BindingCollection, ...],
    child_bindings: Mapping[str, BindingCollection],
) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
    """Attach a declared wrapper and return controls followed by child outputs."""
    if function.registers is None:
        raise ValueError("Feedback requires a named register function")
    control_ports = function.registers[: len(control_bindings)]
    names = tuple(register.name for register in control_ports)
    if names and any(name in child_bindings for name in names):
        raise ValueError("Feedback control bindings overlap child input names")
    actual = dict(zip(names, control_bindings, strict=True))
    outputs = builder.call_dict(function.name, {**child_bindings, **actual})
    returned_controls = tuple(outputs.pop(name) for name in names)
    return returned_controls, tuple(outputs.values())


def classical_bit_feedback_function(
    name: str,
    registers: tuple[Register, ...],
    circuit: CircuitIR,
    controls: tuple[tuple[tuple[int, ...], tuple[bool, ...]], ...],
) -> FunctionRecord:
    """Condition a coherent bit-gate body on ordered scalar or array CBits.

    Each control element is read as an existing bool snapshot. Its owned CBit
    successor returns to the caller; no new classical slot is allocated.
    """
    if any(
        op.name in {"measure", "reset", "classical_xor"}
        or is_classical_control_instruction(op)
        for op in circuit.instructions
    ):
        raise ValueError("Typed bit feedback requires an unconditional coherent body")
    required = tuple(value for _, values in controls for value in values)
    if any(
        any(type(size) is not int or size < 1 for size in shape)
        or len(values) != prod(shape)
        or any(type(value) is not bool for value in values)
        for shape, values in controls
    ):
        raise ValueError("Feedback values must match the concrete CBit control shapes")
    signature = classical_control_signature(
        Signature(registers),
        tuple((classical_bit_type(), shape) for shape, _ in controls),
    )
    child = bit_gate_function(name, registers, circuit)
    block = child.blocks[0]
    assert block.return_values is not None
    next_index = count(
        1
        + max(
            value.index
            for value in (
                *block.arguments,
                *(r for op in block.operations for r in op.results),
            )
        )
    )
    inputs = tuple(ValueRef("entry", next(next_index), "register") for _ in required)
    carried = tuple(ValueRef("entry", next(next_index), "register") for _ in required)
    snapshots = tuple(ValueRef("entry", next(next_index), "bool") for _ in required)
    reads = (
        Operation("read_classical_register", (before,), (after, snapshot))
        for before, after, snapshot in zip(inputs, carried, snapshots, strict=True)
    )
    clause = tuple(
        ConditionRef(snapshot, value)
        for snapshot, value in zip(snapshots, required, strict=True)
    )
    body = (
        (
            replace(op, condition_clauses=(clause,))
            if clause
            and op.opcode not in {"split_register", "join_register", "constant"}
            else op
        )
        for op in block.operations
    )
    return replace(
        child,
        blocks=(
            replace(
                block,
                arguments=(*inputs, *block.arguments),
                operations=(*reads, *body),
                return_values=(*carried, *block.return_values),
            ),
        ),
        registers=tuple(signature),
    )


def classical_cz_word_function(name: str) -> FunctionRecord:
    """Forward one owned CBit and apply CZ when its shared snapshot is one."""
    bit = quantum_bit_type()
    return classical_bit_feedback_function(
        name,
        (Register("q1", bit), Register("q2", bit)),
        CircuitIR(2, (Instruction("cz", (0, 1)),)),
        (((), (True,)),),
    )
