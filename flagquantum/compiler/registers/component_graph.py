# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Named traversal and rebuilding adapted from Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Traverse and rebuild typed calls using the authoritative Program SSA."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace

import numpy as np

from ...core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .component_bindings import create_register_bindings
from .component_builder import BindingCollection, ComponentProgramBuilder
from .component_schedule import local_operation_order


def _call_entry(record: ProgramRecord, entry: str) -> FunctionRecord:
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid component graph: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown component entry {entry!r}")
    function = functions[entry]
    if (
        record.physical_context is not None
        or function.registers is None
        or len(function.blocks) != 1
    ):
        raise ValueError("component traversal requires one typed entry block")
    block = function.blocks[0]
    if block.terminator != "return" or block.edges or block.return_values is None:
        raise ValueError("component traversal requires explicit returned values")
    if any(
        op.opcode != "call" or functions[op.callee or ""].registers is None
        for op in block.operations
    ):
        raise ValueError("component traversal requires typed calls in the entry")
    return function


def _collections(
    registers: Iterable[Register], values: Iterable[RegisterBinding]
) -> dict[str, BindingCollection]:
    bindings = iter(values)
    result: dict[str, BindingCollection] = {}
    for register in registers:
        array = np.empty(register.shape, dtype=object)
        for index in register.all_idxs():
            array[index] = next(bindings)
        result[register.name] = array if register.shape else array[()]
    return result


def _defined_bindings(
    registers: Iterable[Register], values: Iterable[ValueRef]
) -> dict[str, BindingCollection]:
    identities = iter(values)
    return {
        register.name: create_register_bindings(identities.__next__, register)
        for register in registers
    }


def _elements(collections: Iterable[BindingCollection]) -> Iterator[RegisterBinding]:
    for collection in collections:
        for binding in np.asarray(collection).flat:
            assert isinstance(binding, RegisterBinding)
            yield binding


def _iter_calls(
    function: FunctionRecord, functions: Mapping[str, FunctionRecord]
) -> Iterator[tuple[int, dict[str, BindingCollection], tuple[BindingCollection, ...]]]:
    assert function.registers is not None
    block = function.blocks[0]
    inputs = _defined_bindings(Signature(function.registers).lefts(), block.arguments)
    known = {binding.value: binding for binding in _elements(inputs.values())}
    for position in local_operation_order(function, functions, logical_graph=True):
        operation = block.operations[position]
        callee = functions[operation.callee or ""]
        assert callee.registers is not None
        signature = Signature(callee.registers)
        predecessors = _collections(
            signature.lefts(), (known[value] for value in operation.operands)
        )
        outputs = _defined_bindings(signature.rights(), operation.results)
        known.update(
            (binding.value, binding) for binding in _elements(outputs.values())
        )
        yield position, predecessors, tuple(outputs.values())


def iter_component_calls(
    record: ProgramRecord, entry: str = "main"
) -> Iterator[tuple[int, dict[str, BindingCollection], tuple[BindingCollection, ...]]]:
    """Yield source-ordered typed calls with predecessor and output collections.

    Bindings refer to input Program identities. Input dictionary keys describe
    the callee ports, while each binding retains its predecessor's metadata.
    Invalid modules and entries outside the finite typed-call profile raise
    ValueError before the first result.
    """
    function = _call_entry(record, entry)
    yield from _iter_calls(function, {f.name: f for f in record.functions})


def copy_component_program(record: ProgramRecord, entry: str = "main") -> ProgramRecord:
    """Rebuild one typed call entry in visitation order with fresh SSA bindings.

    Definitions and record metadata are retained. Invalid modules and entries
    outside the finite typed-call profile raise ValueError before construction.
    """
    function = _call_entry(record, entry)
    assert function.registers is not None
    functions = {f.name: f for f in record.functions}
    builder = ComponentProgramBuilder(
        Signature(function.registers),
        (f for f in record.functions if f.name != entry),
        name=entry,
    )
    block = function.blocks[0]
    mapped = dict(zip(block.arguments, _elements(builder.inputs.values()), strict=True))
    for position, inputs, outputs in _iter_calls(function, functions):
        callee = functions[block.operations[position].callee or ""]
        assert callee.registers is not None
        arguments = _collections(
            Signature(callee.registers).lefts(),
            (mapped[b.value] for b in _elements(inputs.values())),
        )
        successors = builder.call_dict(callee.name, arguments)
        mapped.update(
            (old.value, new)
            for old, new in zip(
                _elements(outputs), _elements(successors.values()), strict=True
            )
        )
    assert block.return_values is not None
    returned = _collections(
        Signature(function.registers).rights(),
        (mapped[value] for value in block.return_values),
    )
    rebuilt = builder.finalize(returned).functions[0].blocks[0]
    rebuilt = replace(
        block,
        arguments=rebuilt.arguments,
        operations=rebuilt.operations,
        return_values=rebuilt.return_values,
    )
    result = replace(
        record,
        functions=tuple(
            replace(f, blocks=(rebuilt,)) if f.name == entry else f
            for f in record.functions
        ),
    )
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid rebuilt component graph: {diagnostics}")
    return result


def _namespace_helpers(module: ProgramRecord, name: str) -> ProgramRecord:
    """Keep one entry name and isolate its locally constructed definitions."""
    names = {
        f.name: name if f.name == name else name + "_" + f.name
        for f in module.functions
    }
    return replace(
        module,
        functions=tuple(
            replace(
                f,
                name=names[f.name],
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            replace(op, callee=names[op.callee]) if op.callee else op
                            for op in block.operations
                        ),
                    )
                    for block in f.blocks
                ),
            )
            for f in module.functions
        ),
    )
