# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Control propagation adapted from Qualtran v0.7.0 Controlled.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Thread a positive control through stored typed calls and exact specializations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from .component_builder import BindingCollection, ComponentProgramBuilder
from .component_expansion import _has_call_decomposition
from .component_graph import _call_entry, _collections, _elements
from .component_schedule import local_operation_order
from .control.component_predicate import (
    _bit_function,
    control_predicate_program,
    validate_control_ports,
)
from .control.control_predicate import ControlRegister

ControlAdder = Callable[
    [ComponentProgramBuilder, BindingCollection, dict[str, BindingCollection]],
    tuple[BindingCollection, tuple[BindingCollection, ...]],
]


def _control_names(registers: tuple[Register, ...], count: int) -> tuple[str, ...]:
    """Choose source control names using only the underlying data signature."""
    names = {register.name for register in registers}
    if count == 1 and "ctrl" not in names:
        return ("ctrl",)
    index = 1 if "ctrl" in names else 0
    result: list[str] = []
    while len(result) < count:
        index += 1
        if f"ctrl{index}" not in names:
            result.append(f"ctrl{index}")
    return tuple(result)


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
    (control_name,) = _control_names(function.registers, 1)
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


def control_component_via_predicate(
    record: ProgramRecord,
    controlled: Mapping[str, ControlAdder],
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...],
    entry: str = "main",
) -> ProgramRecord:
    """Wrap a stored hierarchy using the pinned ControlledViaAnd call chain.

    Control names are caller supplied and precede the original data ports.
    An explicit entry adder invokes the actual positive-control callback,
    including directional ports; otherwise a THRU hierarchy is derived.
    Directional output names must match the source callback zip order.
    Integer words are row-major and MSB-first. A single QBit is scalar or shape
    (1,). Multiple bits produce n-1 owned predicate outputs, call the positive
    body once, then measure-erase and release them. Specialized adders own their
    private workspace and must preserve the predicate. Invalid controls, name
    collisions and incompatible data signatures raise ValueError.
    """
    function = _call_entry(record, entry)
    count = validate_control_ports(controls, patterns)
    assert function.registers is not None
    if {r.name for r in controls} & {r.name for r in function.registers}:
        raise ValueError("control names conflict with component data ports")
    if count == 1 and (
        len(controls) != 1
        or controls[0].dtype != quantum_bit_type()
        or controls[0].shape not in ((), (1,))
    ):
        raise ValueError("single control requires a scalar or shape-(1,) QBit")
    adder = controlled.get(entry)
    if adder is not None:
        rights = tuple(r.name for r in Signature(function.registers).rights())
        zipped = tuple(dict.fromkeys(r.name for r in function.registers[: len(rights)]))
        if zipped != rights:
            raise ValueError(
                "source predicate callback cannot pair directional output names"
            )
        positive = record
    else:
        positive = control_component_hierarchy(record, controlled, entry)
    names = {f.name for f in positive.functions}
    index = 0
    prefix = "__fq_predicate_0_"
    while any(name.startswith(prefix) for name in names):
        index += 1
        prefix = f"__fq_predicate_{index}_"
    definitions = [f for f in positive.functions if f.name != entry]
    if adder is None:
        derived = next(f for f in positive.functions if f.name == entry)
        derived = replace(derived, name=prefix + "positive")
        definitions.insert(0, derived)
        adder = _stored_control_adder(derived)
    if count > 1:
        inner_controls = tuple(
            Register(f"ctrl_{i}", r.dtype, r.shape) for i, r in enumerate(controls)
        )
        for uncompute, suffix in ((False, "compute"), (True, "erase")):
            definitions.extend(
                control_predicate_program(
                    inner_controls,
                    patterns,
                    name=prefix + suffix,
                    uncompute=uncompute,
                ).functions
            )
    elif patterns[0].bits() == (0,):
        definitions.append(
            _bit_function(
                prefix + "x",
                (Register("q", quantum_bit_type()),),
                CircuitIR(1, (Instruction("x", (0,)),)),
            )
        )
    builder = ComponentProgramBuilder(
        Signature((*controls, *function.registers)), definitions, name=entry
    )
    data = {
        r.name: builder.inputs[r.name] for r in Signature(function.registers).lefts()
    }
    if count == 1:
        q = np.asarray(builder.inputs[controls[0].name]).flat[0]
        negative = patterns[0].bits() == (0,)
        if negative:
            q = builder.call_dict(prefix + "x", {"q": q})["q"]
    else:
        predicate = builder.call_dict(
            prefix + "compute",
            {f"ctrl_{i}": builder.inputs[r.name] for i, r in enumerate(controls)},
        )
        q = predicate.pop("target")
    q, successors = adder(builder, q, data)
    outputs = dict(
        zip(
            (r.name for r in Signature(function.registers).rights()),
            successors,
            strict=True,
        )
    )
    if count == 1:
        if negative:
            q = builder.call_dict(prefix + "x", {"q": q})["q"]
        outputs[controls[0].name] = (
            np.array([q], dtype=object) if controls[0].shape else q
        )
    else:
        predicate["target"] = q
        restored = builder.call_dict(prefix + "erase", predicate)
        outputs.update((r.name, restored[f"ctrl_{i}"]) for i, r in enumerate(controls))
    return replace(
        record, canonical_payload=None, functions=builder.finalize(outputs).functions
    )
