# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Classical control semantics adapted from Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/classically_controlled.py

"""Owned classical controls composed with existing conditional quantum gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import count, product
from math import prod

from ...core.classical_control import is_classical_control_instruction
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import ConditionRef, FunctionRecord, Operation, ValueRef
from ...core.ir.program.register_types import classical_bit_type, quantum_bit_type
from ...core.ir.program.register_verifier import executable_classical_register_width
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from .classical_word_storage import classical_word_bits
from .component_builder import BindingCollection, ComponentProgramBuilder
from .component_control import _control_names
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
    names = _control_names(tuple(child), len(controls))
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
    child = bit_gate_function(name, registers, circuit)
    return _feedback_component_function(child, controls, {})


def _feedback_component_function(
    child: FunctionRecord,
    controls: tuple[tuple[tuple[int, ...], tuple[int | bool, ...]], ...],
    callees: Mapping[str, str],
    control_types: tuple[RegisterType, ...] = (),
) -> FunctionRecord:
    """Thread owned classical elements and conjoin their snapshots on primitives."""
    dtypes = control_types or (classical_bit_type(),) * len(controls)
    if len(dtypes) != len(controls):
        raise ValueError("Feedback needs one classical dtype per control group")
    for dtype in dtypes:
        executable_classical_register_width(dtype)
    if any(
        any(type(size) is not int or size < 0 for size in shape)
        or len(values) != prod(shape)
        or (dtype == classical_bit_type() and any(type(v) is not bool for v in values))
        for dtype, (shape, values) in zip(dtypes, controls, strict=True)
    ):
        raise ValueError("Feedback values must match the concrete CBit control shapes")
    words = tuple(
        (dtype, classical_word_bits(dtype, value))
        for dtype, (_, values) in zip(dtypes, controls, strict=True)
        for value in values
    )
    required = tuple(bit for _, bits in words for bit in bits)
    alternatives = tuple(
        (
            (bits, (1,) * len(bits))
            if dtype.encoding == "ones_complement" and not any(bits)
            else (bits,)
        )
        for dtype, bits in words
    )
    patterns = tuple(
        tuple(bit for bits in combination for bit in bits)
        for combination in product(*alternatives)
    )
    if child.registers is None or len(child.blocks) != 1:
        raise ValueError("Feedback requires a single typed register block")
    signature = classical_control_signature(
        Signature(child.registers),
        tuple(
            (dtype, shape) for dtype, (shape, _) in zip(dtypes, controls, strict=True)
        ),
    )
    block = child.blocks[0]
    if block.return_values is None or block.edges or block.terminator != "return":
        raise ValueError("Feedback requires an explicit finite return")
    if any(
        op.opcode
        in {
            "measure",
            "measure_register",
            "reset",
            "parity_register",
            "discard_classical_register",
        }
        for op in block.operations
    ):
        raise ValueError("Feedback requires a coherent body without classical writes")
    next_index = count(
        1
        + max(
            (
                value.index
                for value in (
                    *block.arguments,
                    *(r for op in block.operations for r in op.results),
                )
            ),
            default=-1,
        )
    )
    inputs = tuple(ValueRef(block.name, next(next_index), "register") for _ in words)
    carried = inputs
    operations: list[Operation] = []
    snapshots = tuple(ValueRef(block.name, next(next_index), "bool") for _ in required)
    # Structural operations remain unconditional, including private work lifetimes.
    structural = {
        "call",
        "split_register",
        "join_register",
        "constant",
        "allocate_register",
        "free_register",
        "read_classical_register",
        "cast_classical_register",
        "bool_zero",
        "bool_one",
        "bool_xor",
    }
    if any(op.opcode not in structural for op in block.operations):
        carried = tuple(
            ValueRef(block.name, next(next_index), "register") for _ in words
        )
        offset = 0
        for before, after, (_, bits) in zip(inputs, carried, words, strict=True):
            end = offset + len(bits)
            operations.append(
                Operation(
                    "read_classical_register",
                    (before,),
                    (after, *snapshots[offset:end]),
                )
            )
            offset = end
    guards = tuple(
        tuple(
            ConditionRef(snapshot, bool(value))
            for snapshot, value in zip(snapshots, pattern, strict=True)
        )
        for pattern in patterns
    )
    for op in block.operations:
        if op.opcode == "call":
            returned = tuple(
                ValueRef(block.name, next(next_index), "register") for _ in words
            )
            operations.append(
                replace(
                    op,
                    callee=callees[op.callee or ""],
                    operands=(*carried, *op.operands),
                    results=(*returned, *op.results),
                )
            )
            carried = returned
        elif required and op.opcode not in structural:
            clauses = tuple(
                tuple(
                    sorted(
                        (*clause, *guard),
                        key=lambda item: (
                            item.value.scope,
                            item.value.index,
                            item.equals,
                        ),
                    )
                )
                for clause in (op.condition_clauses or ((),))
                for guard in guards
            )
            clauses = tuple(
                sorted(
                    clauses,
                    key=lambda clause: tuple(
                        (item.value.scope, item.value.index, item.equals)
                        for item in clause
                    ),
                )
            )
            operations.append(replace(op, condition_clauses=clauses))
        else:
            operations.append(op)
    return replace(
        child,
        blocks=(
            replace(
                block,
                arguments=(*inputs, *block.arguments),
                operations=tuple(operations),
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
