# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _BookkeepingBloq.get_ctrl_system.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/_bookkeeping_bloq.py

"""Pass controls unchanged through verified lifetime and register rebindings."""

from collections.abc import Mapping, Sequence
from typing import TypeVar

from ....core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import ControlAdder
from .component_extension import StructuredControlAdder

_Controls = TypeVar("_Controls", bound=Sequence[BindingCollection])


def bookkeeping_scalar_control_adder(function: FunctionRecord) -> ControlAdder:
    """Adapt a verified bookkeeping leaf to scalar positive-control propagation."""
    _, structured = bookkeeping_component_control_system(
        ProgramRecord((function,)), function.name
    )

    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        controls, outputs = structured(builder, (control,), values)
        return controls[0], outputs

    return add


def call_with_unchanged_controls(
    builder: ComponentProgramBuilder,
    callee: str,
    controls: _Controls,
    values: Mapping[str, object],
) -> tuple[_Controls, tuple[BindingCollection, ...]]:
    """Call an already selected component and retain its control container."""
    return controls, builder.call_tuple(callee, values)


def bookkeeping_component_control_system(
    record: ProgramRecord, entry: str = "main"
) -> tuple[ProgramRecord, StructuredControlAdder]:
    """Return the stored bookkeeping body and its control-pass-through callback.

    Reachable typed definitions may be declarations or contain only calls,
    word allocation/free and split/join operations. Declared components retain
    their signatures and require a body before physical binding. Unrelated
    definitions are verified but need not be bookkeeping. Quantum/classical
    computation, untyped bodies, physical contexts and invalid graphs raise
    ValueError. Controls are returned as the original tuple; no control descriptor, predicate or gate is added. The
    existing builder owns input consumption and errors. Dirty restoration and
    allocated-output transfer remain obligations of the caller and binder.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid bookkeeping component graph: {diagnostics}")
    functions = {f.name: f for f in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown bookkeeping component entry {entry!r}")
    if record.physical_context is not None:
        raise ValueError("bookkeeping control requires a logical component graph")
    pending = [entry]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        function = functions[name]
        if function.registers is None:
            raise ValueError("bookkeeping control requires finite typed bodies")
        for block in function.blocks:
            for operation in block.operations:
                if operation.opcode == "call":
                    pending.append(operation.callee or "")
                elif operation.opcode not in {
                    "allocate_register",
                    "free_register",
                    "split_register",
                    "join_register",
                }:
                    raise ValueError(
                        "bookkeeping control cannot ignore computational operations"
                    )

    def add(
        builder: ComponentProgramBuilder,
        controls: tuple[BindingCollection, ...],
        values: dict[str, BindingCollection],
    ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
        return call_with_unchanged_controls(builder, entry, controls, values)

    return record, add
