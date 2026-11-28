# Copyright 2023-2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled modular addition/subtraction.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic

"""Retain source control specializations across modular arithmetic calls."""

from dataclasses import replace

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ..arithmetic.addition_component import addition_component_program
from ..arithmetic.additional_inverse_control_component import (
    additional_inverse_controlled_addition_program,
    controlled_inverse_addition_program,
)
from ..arithmetic.constant_addition_component import constant_addition_program
from ..arithmetic.controlled_addition_component import controlled_addition_program
from ..arithmetic.nested_controlled_constant_component import (
    nested_controlled_constant_addition_program,
)
from ..arithmetic.specialized_controlled_addition_component import (
    specialized_controlled_addition_program,
)
from ..bits.bitwise_component import bitwise_word_program
from ..bits.bitwise_component_control import (
    _controlled_x,
    controlled_bitwise_word_program,
)
from ..comparison._component_gates import comparison_bit_functions
from ..comparison.controlled_component import controlled_greater_program
from ..comparison.controlled_linear_component import (
    controlled_linear_greater_program,
    nested_controlled_greater_program,
)
from ..comparison.linear_component import linear_greater_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import (
    ControlAdder,
    component_program_control_adder,
    control_component_program,
)
from ..component_definition_match import matches_component_definition
from ..control.component_bookkeeping import bookkeeping_scalar_control_adder
from .controlled_modular_addition_component import controlled_modular_addition_program
from .controlled_modular_subtraction_component import (
    controlled_modular_subtraction_program,
)
from .modular_addition_component import modular_addition_program
from .modular_subtraction_component import modular_subtraction_program


def _specialized_addition_adder(program: ProgramRecord) -> ControlAdder:
    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        outputs = builder.call_program(
            program,
            {
                "ctrl": np.array([control, values["ctrl"]], dtype=object),
                "a": values["a"],
                "b": values["b"],
            },
        )
        controls = np.asarray(outputs["ctrl"])
        return controls[0], (controls[1], outputs["a"], outputs["b"])

    return add


def _control_calls(
    record: ProgramRecord, owners: tuple[tuple[ProgramRecord, ControlAdder], ...]
) -> ProgramRecord:
    x_gate, cx_gate = comparison_bit_functions()
    functions = {function.name: function for function in record.functions}
    adders: dict[str, ControlAdder] = {}
    for operation in record.functions[0].blocks[0].operations:
        assert operation.callee is not None
        name = operation.callee
        if name in adders:
            continue
        # Unequal ports disprove literal equality; equal ports still require the full body proof.
        selected = next(
            (
                callback
                for original, callback in owners
                if functions[name].registers == original.functions[0].registers
                and matches_component_definition(record, name, original)
            ),
            None,
        )
        if selected is not None:
            adders[name] = selected
        elif functions[
            name
        ].registers == x_gate.registers and matches_component_definition(
            record, name, ProgramRecord((x_gate,))
        ):
            adders[name] = _controlled_x
        else:
            adders[name] = bookkeeping_scalar_control_adder(functions[name])
    return control_component_program(
        replace(
            record,
            functions=(
                *record.functions,
                replace(cx_gate, name="bitwise_controlled_x"),
            ),
        ),
        adders,
    )


def additional_modular_addition_program(
    dtype: RegisterType,
    modulus: int,
    *,
    control_value: int | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Add one positive scalar control to ModAdd or CModAdd and its inverse.

    None selects ordinary ModAdd and requires its exact Montgomery descriptor.
    Zero/one selects the original CModAdd polarity. Words remain MSB first;
    bookkeeping, Add-to-CAdd and three-call CAdd reduction follow source owners.
    Actual source-refused controlled comparator leaves remain unbound.
    """
    width = executable_register_width(dtype)
    if control_value is None:
        if dtype != montgomery_register_type(width, modulus):
            raise ValueError(
                "Ordinary ModAdd requires its source Montgomery descriptor"
            )
        record = modular_addition_program(width, modulus, adjoint=adjoint)
        word, wide = montgomery_register_type(width), montgomery_register_type(
            width + 1
        )
        addition = addition_component_program(
            width + 1, left_dtype=wide, right_dtype=wide, adjoint=adjoint
        )
        controlled_add = component_program_control_adder(
            controlled_inverse_addition_program(wide, wide)
            if adjoint
            else controlled_addition_program(wide, wide)
        )
        comparison = linear_greater_program(width, adjoint=adjoint)
        controlled_compare = controlled_linear_greater_program(width, adjoint=adjoint)
    else:
        record = controlled_modular_addition_program(
            dtype, modulus, control_value=control_value, adjoint=adjoint
        )
        word, wide = unsigned_register_type(width), unsigned_register_type(width + 1)
        addition = controlled_addition_program(
            word, wide, control_value=control_value, adjoint=adjoint
        )
        controlled_add = (
            component_program_control_adder(
                additional_inverse_controlled_addition_program(
                    word, wide, control_value=control_value
                )
            )
            if adjoint
            else _specialized_addition_adder(
                specialized_controlled_addition_program(
                    word, wide, values=(1, control_value)
                )
            )
        )
        comparison = controlled_greater_program(
            word, control_value=control_value, adjoint=adjoint
        )
        controlled_compare = nested_controlled_greater_program(
            word, control_value=control_value, adjoint=adjoint
        )
    pattern = (-modulus) % (1 << (width + 1))
    owners = (
        (addition, controlled_add),
        (comparison, component_program_control_adder(controlled_compare)),
        (
            constant_addition_program(wide, pattern, adjoint=adjoint),
            component_program_control_adder(
                constant_addition_program(
                    wide, pattern, adjoint=adjoint, control_value=1
                )
            ),
        ),
        (
            constant_addition_program(word, modulus, adjoint=adjoint, control_value=1),
            component_program_control_adder(
                nested_controlled_constant_addition_program(
                    word, modulus, adjoint=adjoint
                )
            ),
        ),
    )
    return _control_calls(record, owners)


def additional_modular_subtraction_program(
    dtype: RegisterType,
    modulus: int,
    *,
    control_value: int | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Control the five source complement/load/subtract/unload calls literally.

    The optional inner control selects CModSub and retains its original polarity.
    An exact inverse reverses the same shell and uses the exact modular-addition
    inverse. Source-refused descendants are retained only at their actual leaves.
    """
    if type(adjoint) is not bool:
        raise ValueError("Modular subtraction direction must be Boolean")
    width = executable_register_width(dtype)
    record = (
        modular_subtraction_program(dtype, modulus, adjoint=adjoint)
        if control_value is None
        else controlled_modular_subtraction_program(
            dtype, modulus, control_value=control_value, adjoint=adjoint
        )
    )
    load = constant_addition_program(dtype, modulus + 1)
    unload = constant_addition_program(dtype, modulus + 1, adjoint=True)
    complement = bitwise_word_program(dtype, "bitwise_not")
    addition = (
        modular_addition_program(width, modulus)
        if control_value is None
        else controlled_modular_addition_program(
            dtype, modulus, control_value=control_value
        )
    )
    inverse_addition = (
        modular_addition_program(width, modulus, adjoint=True)
        if control_value is None
        else controlled_modular_addition_program(
            dtype, modulus, control_value=control_value, adjoint=True
        )
    )
    addition_dtype = (
        montgomery_register_type(width, modulus) if control_value is None else dtype
    )
    return _control_calls(
        record,
        (
            (
                complement,
                component_program_control_adder(
                    controlled_bitwise_word_program(dtype, "bitwise_not")
                ),
            ),
            (
                load,
                component_program_control_adder(
                    constant_addition_program(dtype, modulus + 1, control_value=1)
                ),
            ),
            (
                unload,
                component_program_control_adder(
                    constant_addition_program(
                        dtype, modulus + 1, control_value=1, adjoint=True
                    )
                ),
            ),
            (
                inverse_addition if adjoint else addition,
                component_program_control_adder(
                    additional_modular_addition_program(
                        addition_dtype,
                        modulus,
                        control_value=control_value,
                        adjoint=adjoint,
                    )
                ),
            ),
        ),
    )
