# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Wiring adapted from Qualtran v0.7.0 cbloq_to_quimb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_quimb.py

"""Wire an already selected typed SSA graph into existing numerical nodes."""

from collections.abc import Callable, Iterable, Mapping

import torch

from ...core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .models import TensorNetworkNode
from .register_identity import register_identity_nodes


def component_tensor_nodes(
    record: ProgramRecord,
    order: tuple[int, ...],
    supply: Callable[
        [
            FunctionRecord,
            Mapping[str, tuple[tuple[int, ...], ...]],
            Mapping[str, tuple[tuple[int, ...], ...]],
        ],
        Iterable[TensorNetworkNode],
    ],
    like: torch.Tensor,
    entry: str = "main",
) -> tuple[
    tuple[TensorNetworkNode, ...],
    dict[str, tuple[tuple[int, ...], ...]],
    dict[str, tuple[tuple[int, ...], ...]],
]:
    """Attach callee tensors to SSA edges and preserve unused external bits.

    Compiler supplies an already selected entry and its typed call visitation
    order. The callback owns each leaf's numerical tensors, including pure-state
    projections. Integer labels are local numerical edges, never physical wires.
    Output and input maps use row-major words and MSB-first bit order and feed
    contract_register_tensors directly. No graph transformation, decomposition,
    backend selection or resource estimation happens here.

    Only one verified typed return block containing unconditional typed calls
    is supported. Classical ports and non-topological/incomplete orders raise
    ValueError before callbacks. Missing suppliers and callback errors propagate.
    Empty networks remain empty; scalar nodes must come from a supplier.
    """

    def attach(
        function: FunctionRecord,
        incoming: Mapping[str, tuple[tuple[int, ...], ...]],
        outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
        fresh_label: Callable[[], int],
    ) -> Iterable[TensorNetworkNode]:
        return supply(function, incoming, outgoing)

    return _component_tensor_nodes(record, order, attach, like, entry, classical=False)


def _component_tensor_nodes(
    record: ProgramRecord,
    order: tuple[int, ...],
    supply: Callable[
        [
            FunctionRecord,
            Mapping[str, tuple[tuple[int, ...], ...]],
            Mapping[str, tuple[tuple[int, ...], ...]],
            Callable[[], int],
        ],
        Iterable[TensorNetworkNode],
    ],
    like: torch.Tensor,
    entry: str,
    *,
    classical: bool,
) -> tuple[
    tuple[TensorNetworkNode, ...],
    dict[str, tuple[tuple[int, ...], ...]],
    dict[str, tuple[tuple[int, ...], ...]],
]:
    """Share SSA wiring and one label namespace across numerical suppliers."""
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid component tensor graph: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown component tensor entry {entry!r}")
    function = functions[entry]
    if (
        record.physical_context is not None
        or function.registers is None
        or len(function.blocks) != 1
    ):
        raise ValueError("component tensors require one typed logical block")
    block = function.blocks[0]
    if block.terminator != "return" or block.edges or block.return_values is None:
        raise ValueError("component tensors require explicit returned values")
    known = set(block.arguments)
    if len(order) != len(block.operations) or set(order) != set(
        range(len(block.operations))
    ):
        raise ValueError("tensor call order must visit every operation exactly once")
    for position in order:
        operation = block.operations[position]
        callee = functions.get(operation.callee or "")
        if (
            operation.opcode != "call"
            or callee is None
            or callee.registers is None
            or operation.parameters
            or operation.condition_clauses
        ):
            raise ValueError("component tensors require unconditional typed calls")
        if not set(operation.operands) <= known:
            raise ValueError("tensor call order must respect SSA dependencies")
        known.update(operation.results)
    for registers in (
        function.registers,
        *(functions[op.callee or ""].registers for op in block.operations),
    ):
        assert registers is not None
        if not classical and any(
            register.dtype.classical_bits for register in registers
        ):
            raise ValueError(
                "pure-state component tensors do not accept classical ports"
            )

    next_label = 0

    def fresh_label() -> int:
        nonlocal next_label
        label = next_label
        next_label += 1
        return label

    def allocate(
        registers: Iterable[Register], values: tuple[ValueRef, ...]
    ) -> dict[ValueRef, tuple[int, ...]]:
        labels = {}
        identities = iter(values)
        for register in registers:
            width = register.bitsize
            assert isinstance(width, int)
            for _ in register.all_idxs():
                labels[next(identities)] = tuple(fresh_label() for _ in range(width))
        return labels

    def ports(
        registers: Iterable[Register],
        values: tuple[ValueRef, ...],
        labels: Mapping[ValueRef, tuple[int, ...]],
    ) -> dict[str, tuple[tuple[int, ...], ...]]:
        identities = iter(values)
        return {
            register.name: tuple(labels[next(identities)] for _ in register.all_idxs())
            for register in registers
        }

    signature = Signature(function.registers)
    inputs = allocate(signature.lefts(), block.arguments)
    outputs = allocate(signature.rights(), block.return_values)
    labels = {**outputs, **inputs}
    incoming = ports(signature.lefts(), block.arguments, inputs)
    outgoing = ports(signature.rights(), block.return_values, outputs)
    nodes: list[TensorNetworkNode] = []
    for position in order:
        operation = block.operations[position]
        callee = functions[operation.callee or ""]
        assert callee.registers is not None
        child = Signature(callee.registers)
        fresh = allocate(child.rights(), operation.results)
        for value, edge in fresh.items():
            if value not in labels:
                labels[value] = edge
        nodes.extend(
            supply(
                callee,
                ports(child.lefts(), operation.operands, labels),
                ports(child.rights(), operation.results, labels),
                fresh_label,
            )
        )
    for value in block.return_values:
        if value in inputs:
            width = len(inputs[value])
            nodes.extend(
                register_identity_nodes(
                    Signature.build(q=width),
                    {"q": (inputs[value],)},
                    {"q": (outputs[value],)},
                    like,
                )
            )
    return tuple(nodes), incoming, outgoing
