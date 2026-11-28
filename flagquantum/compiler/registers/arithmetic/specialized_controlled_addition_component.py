# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _MultiControlledFromSinglyControlled.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Reduce source control arrays around the existing positive CAdd owner."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..control.conjunction_program import and_component_program, conjunction_program
from .controlled_addition_component import controlled_addition_program


def specialized_controlled_addition_program(
    a_dtype: RegisterType,
    b_dtype: RegisterType | None = None,
    *,
    values: tuple[int, ...] = (1, 1),
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain the source compute/CAdd/erase calls under ordered array controls.

    The control array follows source order, and data words are MSB first.
    Conjunction targets, prefix junk and ordinary adder work are private and
    clean on return. Inverse direction selects the exact CAdd inverse while
    preserving the same conjunction owner and all control polarities.
    """
    if len(values) < 2 or any(
        type(value) is not int or value not in (0, 1) for value in values
    ):
        raise ValueError("Specialized addition requires at least two binary controls")
    addition = controlled_addition_program(
        a_dtype, b_dtype, control_value=1, adjoint=adjoint
    )
    target_dtype = a_dtype if b_dtype is None else b_dtype
    prepare, erase = (
        (
            and_component_program(
                (values[0], values[1]), name="control_and", uncompute=inverse
            )
            if len(values) == 2
            else conjunction_program(values, name="control_and", uncompute=inverse)
        )
        for inverse in (False, True)
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("ctrl", quantum_bit_type(), (len(values),)),
                Register("a", a_dtype),
                Register("b", target_dtype),
            )
        )
    )
    predicate = builder.call_program(
        prepare, {"ctrl": builder.inputs["ctrl"]}, entry="control_and"
    )
    added = builder.call_program(
        addition,
        {
            "ctrl": predicate["target"],
            "a": builder.inputs["a"],
            "b": builder.inputs["b"],
        },
    )
    predicate["target"] = added["ctrl"]
    cleared = builder.call_program(erase, predicate, entry="control_and")
    return builder.finalize({"ctrl": cleared["ctrl"], "a": added["a"], "b": added["b"]})
