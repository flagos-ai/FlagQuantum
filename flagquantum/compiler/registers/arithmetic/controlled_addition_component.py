# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CAdd.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/controlled_addition.py

"""Compose controlled addition with owned conjunctions and typed storage views."""

from dataclasses import replace

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_program
from ..component_builder import ComponentProgramBuilder
from ..component_rebindings import join_word_function, split_word_function
from ..control.conjunction_program import and_component_program
from ..layout.component_cast import cast_word_function
from .addition_component import addition_component_program


def controlled_addition_program(
    a_dtype: RegisterType,
    b_dtype: RegisterType | None = None,
    *,
    control_value: int = 1,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain ctrl/a and add a into b on the selected control, MSB first.

    The default b descriptor is a's descriptor. Concrete integer words retain
    their exact dtype. Each input bit owns one temporary conjunction and Add
    owns its carries. All work is measurement-erased and released. A narrower
    signed a follows the source circuit's zero extension. The exact adjoint
    reverses typed calls using the shared measured Add adjoint and And erasure.
    Invalid polarities, directions and nonconcrete integer words raise ValueError.
    """
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Controlled addition control value must be zero or one")
    if type(adjoint) is not bool:
        raise ValueError("Controlled addition direction requires a boolean")
    target_dtype = a_dtype if b_dtype is None else b_dtype
    width = executable_register_width(a_dtype)
    addition = addition_component_program(
        width,
        executable_register_width(target_dtype),
        name="cadd_add",
        left_dtype=a_dtype,
        right_dtype=target_dtype,
    )
    compute = and_component_program((control_value, 1), name="cadd_and")
    erase = and_component_program((control_value, 1), name="cadd_erase", uncompute=True)
    bit, opaque_control = quantum_bit_type(), RegisterType("opaque", 1)
    unsigned = RegisterType("uint", width)
    views = (
        split_word_function("cadd_split_a", a_dtype),
        join_word_function("cadd_join_a", a_dtype),
        split_word_function("cadd_split_ctrl", bit),
        join_word_function("cadd_join_ctrl", opaque_control),
        join_word_function("cadd_join_unsigned", unsigned),
        cast_word_function("cadd_cast", unsigned, a_dtype),
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=bit, a=a_dtype, b=target_dtype),
        (*views, *compute.functions, *erase.functions, *addition.functions),
    )
    a = np.asarray(
        builder.call_dict("cadd_split_a", {"reg": builder.inputs["a"]})["reg"]
    )
    ctrl = np.asarray(
        builder.call_dict("cadd_split_ctrl", {"reg": builder.inputs["ctrl"]})["reg"]
    )[0]
    loaded: list[RegisterBinding] = []
    for index in range(width):
        result = builder.call_dict("cadd_and", {"ctrl": np.array([ctrl, a[index]])})
        ctrl, a[index] = np.asarray(result["ctrl"])
        target = result["target"]
        assert isinstance(target, RegisterBinding)
        loaded.append(target)
    packed = builder.call_dict("cadd_join_unsigned", {"reg": np.array(loaded)})["reg"]
    packed = builder.call_dict("cadd_cast", {"reg": packed})["reg"]
    result = builder.call_dict("cadd_add", {"a": packed, "b": builder.inputs["b"]})
    loaded_bits = np.asarray(
        builder.call_dict("cadd_split_a", {"reg": result["a"]})["reg"]
    )
    for index in reversed(range(width)):
        cleared = builder.call_dict(
            "cadd_erase",
            {"ctrl": np.array([ctrl, a[index]]), "target": loaded_bits[index]},
        )
        ctrl, a[index] = np.asarray(cleared["ctrl"])
    returned_a = builder.call_dict("cadd_join_a", {"reg": a})["reg"]
    outputs = {
        "ctrl": builder.call_dict("cadd_join_ctrl", {"reg": np.array([ctrl])})["reg"],
        "a": returned_a,
        "b": result["b"],
    }
    record = builder.finalize(outputs)
    if not adjoint:
        return record
    inverse_add = addition_component_program(
        width,
        executable_register_width(target_dtype),
        name="cadd_add_inverse",
        left_dtype=a_dtype,
        right_dtype=target_dtype,
        adjoint=True,
    )
    # The shared Add inverse keeps its private carries independent of loaded work.
    definitions = {function.name: function for function in record.functions}
    inverse_views = (
        join_word_function("cadd_join_ctrl_bit", bit),
        split_word_function("cadd_split_ctrl_opaque", opaque_control),
        split_word_function("cadd_split_unsigned", unsigned),
        cast_word_function("cadd_cast_inverse", a_dtype, unsigned),
    )
    for function in (*inverse_views, *inverse_add.functions):
        if function.name in definitions:
            assert definitions[function.name] == function
        definitions[function.name] = function
    return adjoint_component_program(
        replace(record, functions=tuple(definitions.values())),
        {
            "cadd_and": "cadd_erase",
            "cadd_erase": "cadd_and",
            "cadd_add": "cadd_add_inverse",
            "cadd_split_a": "cadd_join_a",
            "cadd_join_a": "cadd_split_a",
            "cadd_split_ctrl": "cadd_join_ctrl_bit",
            "cadd_join_ctrl": "cadd_split_ctrl_opaque",
            "cadd_join_unsigned": "cadd_split_unsigned",
            "cadd_cast": "cadd_cast_inverse",
        },
    )
