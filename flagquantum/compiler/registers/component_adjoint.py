# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Graph reversal adapted from Qualtran v0.7.0 _adjoint_cbloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/adjoint.py

"""Rebuild a typed call graph using explicit specialized adjoint definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ...core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ...core.ir.program.signatures import Signature
from .component_builder import ComponentProgramBuilder
from .component_expansion import _has_call_decomposition
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
    functions = {f.name: f for f in record.functions}
    rebuilt = _adjoint_function(function, functions, adjoints, entry)
    result = replace(
        record,
        canonical_payload=None,
        functions=tuple(rebuilt if f.name == entry else f for f in record.functions),
    )
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid adjoint component graph: {diagnostics}")
    return result


def _adjoint_function(
    function: FunctionRecord,
    functions: Mapping[str, FunctionRecord],
    adjoints: Mapping[str, str],
    name: str,
) -> FunctionRecord:
    assert function.registers is not None
    signature = Signature(function.registers).adjoint()
    block = function.blocks[0]
    for operation in block.operations:
        callee = functions[operation.callee or ""]
        target = adjoints.get(callee.name)
        if target is None or target == name or target not in functions:
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
        signature, (f for f in functions.values() if f.name != name), name=name
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
    return replace(function, name=name, blocks=(rebuilt,), registers=tuple(signature))


def adjoint_component_hierarchy(
    record: ProgramRecord,
    adjoints: Mapping[str, str],
    entry: str = "main",
) -> ProgramRecord:
    """Derive shared stored-body adjoints, stopping at explicit specialized leaves.

    Every unmapped child must supply a typed-call decomposition. Primitive and
    mixed bodies require a caller-provided exact adjoint. The root is derived
    and cannot be mapped. Forward child definitions remain shared and intact.
    Input errors raise ValueError without modifying the immutable input module.
    """
    _call_entry(record, entry)
    if entry in adjoints:
        raise ValueError(
            "hierarchical adjoint derives the entry, which cannot be mapped"
        )
    functions = {f.name: f for f in record.functions}
    resolved = dict(adjoints)
    pending = [(entry, False)]
    generated: list[FunctionRecord] = []
    next_name = 0
    while pending:
        current, ready = pending.pop()
        if current in resolved:
            continue
        function = functions[current]
        if not _has_call_decomposition(function, functions):
            raise ValueError(
                f"missing specialized adjoint for atomic component {current!r}"
            )
        if not ready:
            pending.append((current, True))
            pending.extend(
                (operation.callee or "", False)
                for operation in reversed(function.blocks[0].operations)
                if operation.callee not in resolved
            )
            continue
        name = entry
        if current != entry:
            name = f"__fq_adjoint_{next_name}"
            while name in functions:
                next_name += 1
                name = f"__fq_adjoint_{next_name}"
            next_name += 1
        inverse = _adjoint_function(function, functions, resolved, name)
        resolved[current] = name
        if current == entry:
            functions[entry] = inverse
        else:
            generated.append(inverse)
            functions[name] = inverse
    result = replace(
        record,
        canonical_payload=None,
        functions=(
            *(functions[f.name] for f in record.functions),
            *generated,
        ),
    )
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid hierarchical adjoint graph: {diagnostics}")
    return result
