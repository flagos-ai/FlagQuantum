# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Control propagation adapted from Qualtran v0.7.0 Controlled.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Thread a positive control through stored typed calls and exact specializations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from ...core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from .component_builder import BindingCollection, ComponentProgramBuilder
from .component_expansion import _has_call_decomposition
from .component_graph import _call_entry, _collections, _elements
from .component_schedule import local_operation_order

ControlAdder = Callable[
    [ComponentProgramBuilder, BindingCollection, dict[str, BindingCollection]],
    tuple[BindingCollection, tuple[BindingCollection, ...]],
]


def control_component_program(
    record: ProgramRecord,
    controlled: Mapping[str, ControlAdder],
    entry: str = "main",
) -> ProgramRecord:
    """Prepend one positive QBit and propagate it through specialized child calls.

    The entry must have only THRU ports. Each used child needs an adder that
    consumes the current control and named inputs, then returns its control and
    data successors in the original child's output order. Adders use existing
    builder calls, allowing specialized port renaming and array packing. They
    own controlled semantics, private workspace and measured cleanup. Invalid
    modules, missing adders and incompatible returned ports raise ValueError.
    Adder exceptions propagate. The changed entry clears canonical_payload.
    """
    function = _call_entry(record, entry)
    functions = {f.name: f for f in record.functions}
    rebuilt, added = _control_function(function, functions, controlled, entry)
    result = replace(
        record,
        canonical_payload=None,
        functions=(
            *(rebuilt if f.name == entry else f for f in record.functions),
            *added,
        ),
    )
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid controlled component graph: {diagnostics}")
    return result


def _control_function(
    function: FunctionRecord,
    functions: Mapping[str, FunctionRecord],
    controlled: Mapping[str, ControlAdder],
    name: str,
) -> tuple[FunctionRecord, tuple[FunctionRecord, ...]]:
    assert function.registers is not None
    if any(register.side != RegisterSide.THRU for register in function.registers):
        raise ValueError("controlled component requires only THRU entry ports")
    names = {register.name for register in function.registers}
    control_name = "ctrl"
    if control_name in names:
        index = 2
        while f"ctrl{index}" in names:
            index += 1
        control_name = f"ctrl{index}"
    control = Register(control_name, quantum_bit_type())
    block = function.blocks[0]
    for operation in block.operations:
        if operation.callee not in controlled:
            raise ValueError(f"missing controlled adder for {operation.callee!r}")

    signature = Signature((control, *function.registers))
    builder = ComponentProgramBuilder(
        signature, (f for f in functions.values() if f.name != name), name=name
    )
    current_control = builder.inputs[control_name]
    data = {
        name: value for name, value in builder.inputs.items() if name != control_name
    }
    mapped = dict(zip(block.arguments, _elements(data.values()), strict=True))
    for position in local_operation_order(function, functions, logical_graph=True):
        operation = block.operations[position]
        callee = functions[operation.callee or ""]
        assert callee.registers is not None
        ports = Signature(callee.registers)
        arguments = _collections(
            ports.lefts(), (mapped[value] for value in operation.operands)
        )
        current_control, successors = controlled[callee.name](
            builder, current_control, arguments
        )
        mapped.update(zip(operation.results, _elements(successors), strict=True))
    assert block.return_values is not None
    outputs = _collections(
        signature.rights(),
        (*_elements((current_control,)), *(mapped[v] for v in block.return_values)),
    )
    module = builder.finalize(outputs)
    rebuilt = module.functions[0].blocks[0]
    rebuilt = replace(rebuilt, name=block.name)
    added = tuple(f for f in module.functions[1:] if f.name not in functions)
    return (
        replace(function, name=name, registers=tuple(signature), blocks=(rebuilt,)),
        added,
    )


def _stored_control_adder(function: FunctionRecord) -> ControlAdder:
    assert function.registers is not None
    control_name = function.registers[0].name

    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        outputs = builder.call_dict(function.name, {**values, control_name: control})
        control = outputs.pop(control_name)
        return control, tuple(outputs.values())

    return add


def control_component_hierarchy(
    record: ProgramRecord,
    controlled: Mapping[str, ControlAdder],
    entry: str = "main",
) -> ProgramRecord:
    """Derive shared stored-body controls, stopping at explicit specialized adders.

    Every unmapped child must have a typed-call decomposition with THRU ports.
    Primitive and mixed bodies need an explicit controlled adder. The root is
    derived and cannot be mapped. Original children remain shared and intact.
    Invalid inputs raise ValueError without modifying the immutable module.
    Adder exceptions propagate. The forward canonical payload is cleared.
    """
    _call_entry(record, entry)
    if entry in controlled:
        raise ValueError(
            "hierarchical control derives the entry, which cannot be mapped"
        )
    functions = {f.name: f for f in record.functions}
    resolved = dict(controlled)
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
                f"missing specialized control for atomic component {current!r}"
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
            name = f"__fq_controlled_{next_name}"
            while name in functions:
                next_name += 1
                name = f"__fq_controlled_{next_name}"
            next_name += 1
        derived, added = _control_function(function, functions, resolved, name)
        for helper in added:
            functions[helper.name] = helper
        generated.extend(added)
        resolved[current] = _stored_control_adder(derived)
        if current == entry:
            functions[entry] = derived
        else:
            generated.append(derived)
            functions[name] = derived
    result = replace(
        record,
        canonical_payload=None,
        functions=(*(functions[f.name] for f in record.functions), *generated),
    )
    diagnostics = verify_program_record(result)
    if diagnostics:
        raise ValueError(f"invalid hierarchical controlled graph: {diagnostics}")
    return result
