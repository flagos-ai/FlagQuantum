# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 AdjointWithSpecializedCtrl.get_ctrl_system.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Reverse an actual specialized control-adder graph using exact adjoints."""

from collections.abc import Mapping
from dataclasses import replace

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_hierarchy
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import _control_names
from ..component_graph import _call_entry
from .component_control_selection import ComponentControlSystem
from .component_extension import StructuredControlAdder
from .component_predicate import validate_control_ports
from .control_predicate import ControlRegister


def adjoint_component_control_system(
    record: ProgramRecord,
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...],
    specialize_on: frozenset[int],
    forward_control: ComponentControlSystem,
    inverse_fallback: ComponentControlSystem,
    adjoints: Mapping[str, str],
    entry: str = "main",
) -> tuple[ProgramRecord, StructuredControlAdder]:
    """Push allowed single-bit controls through an exact stored adjoint.

    record is the forward data body. forward_control supplies its real control
    module and adder. Exact adjoints reverse that module and the graph produced
    by its adder. The latter remains a stored local adapter that callers may
    expand once to reproduce the source's add_from inlining. Other requests
    delegate unchanged to inverse_fallback, which owns generic inverse control.
    No numerical execution or new graph representation is introduced.
    """
    count = validate_control_ports(controls, patterns)
    if any(type(value) is not int or value not in (0, 1) for value in specialize_on):
        raise ValueError("specialized adjoint activations must be binary integers")
    if count != 1 or patterns[0].bits()[0] not in specialize_on:
        return inverse_fallback(controls, patterns)
    function = _call_entry(record, entry)
    assert function.registers is not None
    (name,) = _control_names(function.registers, 1)
    selected, forward_adder = forward_control(controls, patterns)
    inverse = adjoint_component_hierarchy(selected, adjoints, entry)
    names = {f.name for f in inverse.functions} | {f.name for f in selected.functions}
    index = 0
    while f"__fq_adjoint_control_{index}" in names:
        index += 1
    adapter_name = f"__fq_adjoint_control_{index}"
    inverse_name = adapter_name + "_inverse"
    while inverse_name in names:
        inverse_name += "_"
    builder = ComponentProgramBuilder(
        Signature((*function.registers, Register(name, quantum_bit_type()))),
        selected.functions,
        name=adapter_name,
    )
    inputs = {
        r.name: builder.inputs[r.name]
        for r in function.registers
        if r.side & RegisterSide.LEFT
    }
    (control,), data = forward_adder(builder, (builder.inputs[name],), inputs)
    outputs = dict(
        zip(
            (r.name for r in Signature(function.registers).rights()), data, strict=False
        )
    )
    outputs[name] = control
    forward = builder.finalize(outputs)
    inverse_root = replace(_call_entry(inverse, entry), name=inverse_name)
    definitions = {
        f.name: f
        for f in (
            *forward.functions,
            *(f for f in inverse.functions if f.name != entry),
            inverse_root,
        )
    }
    reversed_adapter = adjoint_component_hierarchy(
        replace(forward, functions=tuple(definitions.values())),
        {**adjoints, entry: inverse_name},
        adapter_name,
    )
    adapter = next(f for f in reversed_adapter.functions if f.name == adapter_name)
    generated = tuple(
        f
        for f in reversed_adapter.functions
        if f.name not in definitions and f.name != adapter_name
    )
    adapters = tuple(
        replace(
            function,
            blocks=tuple(
                replace(
                    block,
                    operations=tuple(
                        replace(op, callee=entry) if op.callee == inverse_name else op
                        for op in block.operations
                    ),
                )
                for block in function.blocks
            ),
        )
        for function in (adapter, *generated)
    )
    module = replace(inverse, functions=(*inverse.functions, *adapters))

    def add(
        caller: ComponentProgramBuilder,
        added: tuple[BindingCollection, ...],
        values: dict[str, BindingCollection],
    ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
        (ctrl,) = added
        values[name] = ctrl
        result = caller.call_dict(adapter_name, values)
        return (result.pop(name),), tuple(result.values())

    return module, add
