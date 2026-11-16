# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Program width admission follows Qualtran v0.7.0 QubitCount.compute.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py

"""Estimate the logical peak of a native typed register Program."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import replace

from ...core.ir.program import ProgramRecord, verify_program_record
from ...core.ir.program.signatures import Signature
from ..registers.component_expansion import _has_call_decomposition
from ..registers.component_schedule import _function_connections
from .logical_width import query_logical_width
from .symbolic_math import SymbolicInt
from .width_schedule import estimate_connection_width


def query_program_logical_width(
    record: ProgramRecord,
    entry: str = "main",
    *,
    cache: MutableMapping[str, SymbolicInt] | None = None,
    static_costs: Mapping[str, SymbolicInt] | None = None,
) -> SymbolicInt:
    """Query finite typed calls with the same lifetime edges used by scheduling.

    A connected decomposition counts live bystanders and recursive callee
    peaks. Each disconnected component is considered independently. A
    primitive or mixed typed body uses the signature/callee fallback.
    Model widths and physical workspace binding are independent queries.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid register Program: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown Program entry {entry!r}")
    if record.physical_context is not None or functions[entry].registers is None:
        raise ValueError("Logical width requires a typed register Program entry")

    def signature_width(name: str) -> SymbolicInt:
        registers = functions[name].registers
        if registers is None:
            raise ValueError(f"Logical width requires typed ports on {name!r}")
        return Signature(registers).n_qubits()

    def callees(name: str) -> tuple[tuple[str, SymbolicInt], ...]:
        counts = Counter(
            op.callee
            for block in functions[name].blocks
            for op in block.operations
            if op.opcode == "call" and op.callee is not None
        )
        return tuple(counts.items())

    def decomposition(
        name: str, child: Callable[[str], SymbolicInt]
    ) -> SymbolicInt | None:
        function = functions[name]
        if not _has_call_decomposition(function, functions):
            return None
        steps, edges, nodes = _function_connections(
            function, functions, logical_graph=True
        )
        operations = function.blocks[0].operations
        for index, (step, node) in enumerate(zip(steps, nodes, strict=True)):
            if step.kind not in {"input", "output"}:
                callee = operations[node].callee
                assert callee is not None
                steps[index] = replace(step, qubits=child(callee))
        return estimate_connection_width(steps, edges)

    return query_logical_width(
        entry,
        signature_width=signature_width,
        callees=callees,
        decomposition_width=decomposition,
        cache=cache,
        static_costs=static_costs,
    )
