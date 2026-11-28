# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled(AddK), including nested controls.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Control constant loading twice while preserving the source Always(Add)."""

from functools import partial

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import RegisterType
from ..bits.bitwise_component_control import nested_controlled_xor_constant_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import ControlAdder, control_component_program
from .constant_addition_component import _unconditional, constant_addition_program


def nested_controlled_constant_addition_program(
    dtype: RegisterType, pattern: int, *, adjoint: bool = False, control_count: int = 2
) -> ProgramRecord:
    """Add or subtract an encoded pattern under distinct positive source controls.

    Controls remain distinct THRU scalar ports, preceding the MSB-first word.
    The constant word and ordinary adder work are private and clean. Both loads
    use literal nested XOR control; Always(Add) and allocation/free calls retain
    unchanged controls. No grouped-predicate synthesis replaces this recipe.
    """
    if type(control_count) is not int or control_count < 2:
        raise ValueError(
            "Nested constant addition requires at least two positive controls"
        )
    record = (
        constant_addition_program(dtype, pattern, adjoint=adjoint, control_value=1)
        if control_count == 2
        else nested_controlled_constant_addition_program(
            dtype, pattern, adjoint=adjoint, control_count=control_count - 1
        )
    )
    word = nested_controlled_xor_constant_program(
        dtype, pattern, control_count=control_count
    )
    registers = word.functions[0].registers
    assert registers is not None
    control_name = registers[0].name

    def load(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        outputs = builder.call_program(
            word, {control_name: control, **values}, entry=word.functions[0].name
        )
        current = outputs.pop(control_name)
        return current, tuple(outputs.values())

    load_name = record.functions[0].blocks[0].operations[1].callee
    assert load_name is not None
    adders: dict[str, ControlAdder] = {
        name: partial(_unconditional, callee=name)
        for name in ("constant_allocate", "constant_free", "constant_always")
    }
    adders[load_name] = load
    return control_component_program(record, adders)
