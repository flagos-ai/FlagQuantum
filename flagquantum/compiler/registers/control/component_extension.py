# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd.get_ctrl_system.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py

"""Extend stored controlled components by merging clauses at their original body."""

from collections.abc import Callable, Mapping

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import (
    ControlAdder,
    _control_names,
    control_component_via_predicate,
)
from ..component_graph import _call_entry
from .component_predicate import validate_control_ports
from .control_predicate import ControlRegister

StructuredControlAdder = Callable[
    [
        ComponentProgramBuilder,
        tuple[BindingCollection, ...],
        dict[str, BindingCollection],
    ],
    tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]],
]


def extend_component_control_system(
    record: ProgramRecord,
    controlled: Mapping[str, ControlAdder],
    existing_controls: tuple[Register, ...],
    existing_patterns: tuple[ControlRegister, ...],
    additional_controls: tuple[Register, ...],
    additional_patterns: tuple[ControlRegister, ...],
    entry: str = "main",
) -> tuple[ProgramRecord, StructuredControlAdder]:
    """Return a combined wrapper and the source's control-extracting adder.

    The record is the original forward body. Existing names identify the controls
    popped from the adder's mutable input dictionary. Additional descriptor names
    label tuple positions. Generated control names depend only on forward data
    names. New clauses precede old ones. The builder must contain the returned
    module's definitions. The adder returns new controls, then old controls and
    data in declaration order. Predicate and child cleanup use existing owners.

    Invalid descriptors and name collisions raise ValueError. Missing old inputs
    raise KeyError after earlier successful pops, as in the original callback.
    Builder failures retain the existing binding-consumption behavior.
    """
    function = _call_entry(record, entry)
    assert function.registers is not None
    validate_control_ports(existing_controls, existing_patterns)
    validate_control_ports(additional_controls, additional_patterns)
    if {r.name for r in existing_controls} & {r.name for r in function.registers}:
        raise ValueError("existing control names conflict with forward data ports")
    descriptors = (*additional_controls, *existing_controls)
    names = _control_names(function.registers, len(descriptors))
    combined = tuple(
        Register(name, register.dtype, register.shape)
        for name, register in zip(names, descriptors, strict=True)
    )
    module = control_component_via_predicate(
        record, controlled, combined, (*additional_patterns, *existing_patterns), entry
    )
    new_count = len(additional_controls)

    def add(
        builder: ComponentProgramBuilder,
        controls: tuple[BindingCollection, ...],
        values: dict[str, BindingCollection],
    ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
        old = tuple(values.pop(register.name) for register in existing_controls)
        arguments = dict(zip(names, (*controls, *old), strict=False)) | values
        outputs = builder.call_tuple(entry, arguments)
        return outputs[:new_count], outputs[new_count:]

    return module, add
