# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled(Subtract/SubtractFrom).
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Propagate scalar subtraction controls through the original typed children."""

from dataclasses import replace
from functools import partial

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import (
    ControlAdder,
    control_component_hierarchy,
    control_component_via_predicate,
)
from ..component_gate_function import bit_gate_function
from ..control.control_predicate import ControlRegister
from .controlled_addition_component import controlled_addition_program
from .subtraction_declaration import SubtractFromDeclaration, SubtractionDeclaration


def _call_controlled(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
    *,
    record: ProgramRecord,
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_program(record, {"ctrl": control, **values})
    control = outputs.pop("ctrl")
    return control, tuple(outputs.values())


def _preserve(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
    *,
    callee: str,
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    return control, tuple(builder.call_dict(callee, values).values())


def _control_x(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_dict(
        "subtraction_cx", {"ctrl": control, "target": values["q"]}
    )
    return outputs["ctrl"], (outputs["target"],)


def _control_cx(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_dict(
        "subtraction_ccx",
        {
            "ctrl": np.asarray([control, values["ctrl"]], dtype=object),
            "target": values["target"],
        },
    )
    controls = outputs["ctrl"]
    assert isinstance(controls, np.ndarray)
    return controls[0], (controls[1], outputs["target"])


def controlled_subtraction_program(
    a_dtype: RegisterType,
    b_dtype: RegisterType | None = None,
    *,
    subtract_from: bool = False,
    control_value: int = 1,
) -> ProgramRecord:
    """Preserve a/control and conditionally replace b by a-b or b-a, MSB first.

    SubtractFrom requires equal descriptors. Both binary control polarities
    retain source bookkeeping, CAdd specialization and clean measured carries.
    Signed unequal-width Subtract owns its sign prefix and controls the original
    fanout CNOTs through atomic Toffoli calls. Invalid descriptors and control
    values raise ValueError through the existing owners.
    """
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Subtraction control value must be zero or one")
    target = a_dtype if b_dtype is None else b_dtype
    if subtract_from and target != a_dtype:
        raise ValueError("SubtractFrom requires equal operand descriptors")
    record = (
        SubtractFromDeclaration(a_dtype).program()
        if subtract_from
        else SubtractionDeclaration(a_dtype, target).program()
    )
    adders: dict[str, ControlAdder] = {}
    if subtract_from:
        operations = record.functions[0].blocks[0].operations
        for operation, child in zip(
            operations,
            (
                controlled_bitwise_word_program(a_dtype, "bitwise_not"),
                controlled_addition_program(a_dtype),
                controlled_bitwise_word_program(a_dtype, "bitwise_not"),
            ),
            strict=True,
        ):
            assert operation.callee is not None
            adders[operation.callee] = partial(_call_controlled, record=child)
    else:
        adders["subtract_add"] = partial(
            _call_controlled,
            record=controlled_addition_program(
                unsigned_register_type(target.num_qubits)
            ),
        )
        for function in record.functions:
            assert function.registers is not None
            if any(r.side != RegisterSide.THRU for r in function.registers):
                adders[function.name] = partial(_preserve, callee=function.name)
            elif function.name.endswith("subtract_x"):
                adders[function.name] = _control_x
            elif function.name.endswith("fanout_cx"):
                adders[function.name] = _control_cx
        bit = quantum_bit_type()
        leaves = [
            bit_gate_function(
                "subtraction_cx",
                (Register("ctrl", bit), Register("target", bit)),
                CircuitIR(2, (Instruction("cx", (0, 1)),)),
            )
        ]
        if _control_cx in adders.values():
            leaves.append(
                bit_gate_function(
                    "subtraction_ccx",
                    (Register("ctrl", bit, (2,)), Register("target", bit)),
                    CircuitIR(3, (Instruction("ccx", (0, 1, 2)),)),
                )
            )
        record = replace(record, functions=(*record.functions, *leaves))
    if control_value:
        return control_component_hierarchy(record, adders)
    return control_component_via_predicate(
        record,
        adders,
        (Register("ctrl", quantum_bit_type()),),
        (ControlRegister(1, (0,)),),
    )
