# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 Controlled(AutoPartition) call traversal.

"""Specialize existing data partitions around an explicitly controlled child."""

from dataclasses import replace

from ....core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import ControlAdder, control_component_hierarchy
from .auto_partition import AutoPartition
from .component_auto_partition import auto_partition_component_pair


def controlled_auto_partition_pair(
    child: ProgramRecord,
    inverse: ProgramRecord,
    ports: tuple[Register, ...],
    layout: AutoPartition,
    *,
    data_registers: tuple[Register, ...],
) -> tuple[ProgramRecord, ProgramRecord]:
    """Keep data-only Partition calls unconditional while threading one control.

    The exact positive-controlled child pair starts with a scalar QBit and has
    through data ports. Data layout and inverse partition ordering remain owned
    by AutoPartition. The temporary logical child declaration is replaced in
    every call before it is removed; no unresolved execution body is retained.
    """
    registers = child.functions[0].registers
    if (
        not registers
        or not Signature(registers).thru_registers_only
        or registers[0].dtype != quantum_bit_type()
        or registers[0].shape
        or inverse.functions[0].registers != registers
        or not Signature(data_registers).thru_registers_only
        or tuple((r.dtype, r.shape) for r in data_registers)
        != tuple((r.dtype, r.shape) for r in registers[1:])
        or layout.left_only
    ):
        raise ValueError(
            "Controlled partition requires exact through controlled children"
        )
    declaration = ProgramRecord(
        (FunctionRecord("data_child", (), registers=data_registers),)
    )
    forward, backward = auto_partition_component_pair(
        declaration, ports, layout, declaration
    )
    assert backward is not None

    def specialize(template: ProgramRecord, actual: ProgramRecord) -> ProgramRecord:
        declarations = {f.name for f in template.functions if not f.blocks}
        if not declarations:
            raise ValueError("Controlled partition template lost its child declaration")
        control_name = registers[0].name

        def add_child(
            builder: ComponentProgramBuilder,
            control: BindingCollection,
            values: dict[str, BindingCollection],
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            arguments = {
                target.name: values[source.name]
                for source, target in zip(data_registers, registers[1:], strict=True)
            }
            result = builder.call_program(
                actual,
                {control_name: control, **arguments},
                entry=actual.functions[0].name,
            )
            successor = result.pop(control_name)
            return successor, tuple(result.values())

        adders: dict[str, ControlAdder] = {}
        for function in template.functions[1:]:
            if function.name in declarations:
                adders[function.name] = add_child
                continue

            def unchanged(
                builder: ComponentProgramBuilder,
                control: BindingCollection,
                values: dict[str, BindingCollection],
                *,
                name: str = function.name,
            ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
                return control, builder.call_tuple(name, values)

            adders[function.name] = unchanged
        controlled = control_component_hierarchy(template, adders)
        result = replace(
            controlled,
            functions=tuple(
                f for f in controlled.functions if f.name not in declarations
            ),
        )
        if verify_program_record(result):
            raise ValueError(
                "Controlled partition retained an unresolved template call"
            )
        return result

    return specialize(forward, child), specialize(backward, inverse)
