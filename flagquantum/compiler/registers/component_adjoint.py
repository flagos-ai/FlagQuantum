# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Graph reversal adapted from Qualtran v0.7.0 _adjoint_cbloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/adjoint.py

"""Rebuild a typed call graph using explicit specialized adjoint definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ...core.ir.program import ProgramRecord, verify_program_record
from ...core.ir.program.signatures import Signature
from .component_builder import ComponentProgramBuilder
from .component_graph import _call_entry, _collections, _elements
from .component_schedule import local_operation_order


def adjoint_component_program(
    record: ProgramRecord,
    adjoints: Mapping[str, str],
    entry: str = "main",
) -> ProgramRecord:
    """Reverse typed calls and ports using caller-supplied exact adjoint bodies.

    Every used callee needs a target already defined in the module with exactly
    its adjoint signature. The caller owns the semantic validity of that body,
    including specialized measurement and workspace restoration. Invalid modules,
    missing targets and incompatible signatures raise ValueError. Definitions
    remain shared, with fresh SSA identities for the transformed entry. The
    original forward canonical payload is cleared because its semantics change.
    """
    function = _call_entry(record, entry)
    assert function.registers is not None
    signature = Signature(function.registers).adjoint()
    functions = {f.name: f for f in record.functions}
    block = function.blocks[0]
    for operation in block.operations:
        callee = functions[operation.callee or ""]
        target = adjoints.get(callee.name)
        if target is None or target == entry or target not in functions:
            raise ValueError(f"missing adjoint definition for {callee.name!r}")
        assert callee.registers is not None
        target_registers = functions[target].registers
        expected = Signature(callee.registers).adjoint()
        if target_registers is None:
            raise ValueError(f"incompatible adjoint signature for {callee.name!r}")
        actual = Signature(target_registers)
        # Specialized Split/Join adjoints group opposite-side declarations
        # differently. Ordered ports on each side determine call compatibility.
        if tuple(actual.lefts()) != tuple(expected.lefts()) or tuple(
            actual.rights()
        ) != tuple(expected.rights()):
            raise ValueError(f"incompatible adjoint signature for {callee.name!r}")

    builder = ComponentProgramBuilder(
        signature, (f for f in record.functions if f.name != entry), name=entry
    )
    assert block.return_values is not None
    mapped = dict(
        zip(block.return_values, _elements(builder.inputs.values()), strict=True)
    )
    for position in reversed(
        local_operation_order(function, functions, logical_graph=True)
    ):
        operation = block.operations[position]
        callee = functions[operation.callee or ""]
        assert callee.registers is not None
        ports = Signature(callee.registers)
        arguments = _collections(
            ports.rights(), (mapped[value] for value in operation.results)
        )
        successors = builder.call_dict(adjoints[callee.name], arguments)
        mapped.update(
            zip(operation.operands, _elements(successors.values()), strict=True)
        )
    outputs = _collections(
        signature.rights(), (mapped[value] for value in block.arguments)
    )
    rebuilt = builder.finalize(outputs).functions[0].blocks[0]
    rebuilt = replace(rebuilt, name=block.name)
    result = replace(
        record,
        canonical_payload=None,
        functions=tuple(
            (
                replace(f, blocks=(rebuilt,), registers=tuple(signature))
                if f.name == entry
                else f
            )
            for f in record.functions
        ),
    )
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid adjoint component graph: {diagnostics}")
    return result
