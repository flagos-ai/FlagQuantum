# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Local graph ordering adapted from Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Derive call-local physical visitation from existing Program SSA edges."""

from __future__ import annotations

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord, Operation, ValueRef
from ...core.ir.program.register_types import classical_bit_type
from ...core.ir.program.register_verifier import (
    executable_register_width,
    register_port_types,
)
from ...core.ir.program.registers import RegisterSide
from .._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    schedule_width_steps,
)


def local_operation_order(
    function: FunctionRecord,
    functions: Mapping[str, FunctionRecord],
    *,
    logical_graph: bool = False,
) -> tuple[int, ...]:
    """Schedule a verified finite function using connection-created node order.

    Arguments and returns act as dangling nodes. Edges introduce their source
    before their target, following operand traversal and then returned ports.
    Named graph traversal counts owned classical bits as source graph edges.
    Logical traversal classifies only the current component layer. Physical
    scheduling keeps classical quantum width zero and follows nested helpers.
    Isolated operations use declaration order because source frozenset ties
    have no portable order. Call bodies retain their own separate schedule.
    """
    block = function.blocks[0]
    widths = _argument_widths(function, logical_graph)
    producers = dict.fromkeys(block.arguments, -1)
    nodes: dict[int, int] = {}
    edges: list[WidthConnection] = []

    def connect(value: ValueRef, target: int) -> None:
        source = producers[value]
        for node in (source, target):
            if node not in nodes:
                nodes[node] = len(nodes)
        edges.append(WidthConnection(nodes[source], nodes[target], widths[value]))

    for position, operation in enumerate(block.operations):
        for value in operation.operands:
            connect(value, position)
        for _, value in operation.parameters:
            connect(value, position)
        for clause in operation.condition_clauses:
            for condition in clause:
                connect(condition.value, position)
        widths.update(_result_widths(operation, functions, logical_graph))
        producers.update((value, position) for value in operation.results)
    for value in block.return_values or ():
        connect(value, len(block.operations))
    for position in range(len(block.operations)):
        if position not in nodes:
            nodes[position] = len(nodes)
    steps = []
    for node in nodes:
        if node == -1:
            step = WidthStep("input")
        elif node == len(block.operations):
            step = WidthStep("output")
        else:
            step = _operation_step(
                block.operations[node], functions, logical_graph=logical_graph
            )
        steps.append(step)
    schedule = schedule_width_steps(steps, edges, group_components=False)
    ordered_nodes = tuple(nodes)
    return tuple(
        ordered_nodes[index]
        for group in schedule
        for index in group
        if 0 <= ordered_nodes[index] < len(block.operations)
    )


def _argument_widths(
    function: FunctionRecord, logical_graph: bool
) -> dict[ValueRef, int]:
    values = function.blocks[0].arguments
    if function.registers is None:
        return {value: int(value.kind == "qubit") for value in values}
    return dict(
        zip(
            values,
            (
                (
                    int(logical_graph)
                    if dtype == classical_bit_type()
                    else executable_register_width(dtype)
                )
                for dtype in register_port_types(function, RegisterSide.LEFT)
            ),
            strict=True,
        )
    )


def _result_widths(
    operation: Operation,
    functions: Mapping[str, FunctionRecord],
    logical_graph: bool,
) -> dict[ValueRef, int]:
    if operation.opcode in {"join_register", "allocate_register"}:
        assert operation.register_type is not None
        return {
            value: executable_register_width(operation.register_type)
            for value in operation.results
        }
    if operation.opcode == "call":
        callee = functions[operation.callee or ""]
        if callee.registers is not None:
            return dict(
                zip(
                    operation.results,
                    (
                        (
                            int(logical_graph)
                            if dtype == classical_bit_type()
                            else executable_register_width(dtype)
                        )
                        for dtype in register_port_types(callee, RegisterSide.RIGHT)
                    ),
                    strict=True,
                )
            )
    return {value: int(value.kind == "qubit") for value in operation.results}


def _operation_step(
    operation: Operation,
    functions: Mapping[str, FunctionRecord],
    *,
    logical_graph: bool,
) -> WidthStep:
    # Compiler workspace helpers have an actual single atomic body. Classify
    # those calls as their source leaf. Logical graph traversal inspects one
    # body only, preserving a composite wrapper as a neutral component.
    while operation.opcode == "call":
        name = operation.callee or ""
        body = functions[name].blocks[0].operations
        if len(body) != 1:
            break
        operation = body[0]
        if logical_graph:
            break
    if operation.opcode == "allocate_register":
        return WidthStep("allocate")
    if operation.opcode == "free_register":
        return WidthStep("free")
    return WidthStep("component")
