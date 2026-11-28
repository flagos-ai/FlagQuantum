# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Generic control propagation follows Qualtran v0.7.0 _infra/controlled.py.

"""Bind literal THC generic controls to explicit, independently owned children."""

from collections.abc import Mapping
from dataclasses import replace

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import (
    ControlAdder,
    _stored_control_adder,
    control_component_program,
    quantum_control_signature,
)
from ..component_definition_match import matches_component_definition
from ..component_gate_function import bit_gate_function
from ..component_graph import _namespace_helpers
from ..primitives.controlled_swap_component import (
    controlled_swap_program,
    positive_controlled_swap_program,
)
from ..primitives.controlled_z_word_component import (
    controlled_z_word_program,
    positive_controlled_z_word_program,
)
from .component_bookkeeping import bookkeeping_scalar_control_adder


def control_thc_selection_program(
    record: ProgramRecord,
    rotations: tuple[FunctionRecord, ...],
    swaps: Mapping[str, ProgramRecord],
    phases: Mapping[str, tuple[ProgramRecord, tuple[int, ...]]],
    flip: str,
    bookkeeping: tuple[str, ...],
) -> ProgramRecord:
    """Thread ctrl through known calls; retain opaque controlled rotations.

    Roles identify actual imported callees, never structurally equal unknown
    headers. Swap/phase bodies are checked against their source owners. Unknown
    widths retain source-failed child declarations. The angle allocation/free
    pass ctrl unchanged; source rotation headers remain non-executable.
    """
    definitions = list(record.functions)
    controls: dict[str, ControlAdder] = {}

    def add(callee: str, module: ProgramRecord) -> FunctionRecord:
        prefix = f"__fq_thc_positive_{len(controls)}"
        while any(function.name.startswith(prefix) for function in definitions):
            prefix += "_"
        imported = _namespace_helpers(module, prefix)
        definitions.extend(imported.functions)
        function = imported.functions[0]
        controls[callee] = _stored_control_adder(function)
        return function

    def declaration(program: ProgramRecord) -> ProgramRecord:
        ports = program.functions[0].registers
        assert ports is not None
        return ProgramRecord(
            (
                FunctionRecord(
                    program.functions[0].name + "_positive",
                    (),
                    registers=tuple(
                        quantum_control_signature(
                            Signature(ports),
                            ((quantum_bit_type(), ()),),
                        )
                    ),
                ),
            )
        )

    for rotation in rotations:
        if rotation.blocks:
            raise ValueError(
                "Literal THC generic control requires source rotation declarations"
            )
        add(rotation.name, declaration(ProgramRecord((rotation,))))
    for callee, program in swaps.items():
        width = Signature(program.functions[0].registers or ()).get_left("x").bitsize
        if type(width) is int and width > 0:
            if not matches_component_definition(
                program, program.functions[0].name, controlled_swap_program(width)
            ):
                raise ValueError("Literal THC swap differs from its source owner")
            add(callee, positive_controlled_swap_program(width))
        else:
            add(callee, declaration(program))
    for callee, (program, values) in phases.items():
        width = (
            Signature(program.functions[0].registers or ()).get_left("system").bitsize
        )
        if type(width) is int and width > 0:
            if not matches_component_definition(
                program,
                program.functions[0].name,
                controlled_z_word_program(width, values),
            ):
                raise ValueError("Literal THC phase differs from its source owner")
            add(callee, positive_controlled_z_word_program(width, values))
        else:
            add(callee, declaration(program))
    cnot = add(
        flip,
        ProgramRecord(
            (
                bit_gate_function(
                    "positive_flip",
                    tuple(Signature.build(ctrl=1, target=1)),
                    CircuitIR(2, (Instruction("cx", (0, 1)),)),
                ),
            )
        ),
    )

    def add_flip(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        outputs = builder.call_dict(cnot.name, {"ctrl": control, "target": values["q"]})
        return outputs["ctrl"], (outputs["target"],)

    controls[flip] = add_flip
    functions = {function.name: function for function in record.functions}
    for name in bookkeeping:
        controls[name] = bookkeeping_scalar_control_adder(functions[name])
    return control_component_program(
        replace(record, functions=tuple(definitions)), controls
    )
