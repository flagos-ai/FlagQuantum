# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CLinearDepthGreaterThan.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Controlled widened difference comparison with exact measured adder erasure."""

from dataclasses import replace

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.out_of_place_addition_component import out_of_place_addition_program
from ..bits.bitwise_component import bitwise_word_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..control.multi_control_component import multi_control_x_program
from ._linear_words import (
    comparison_width,
    prepare_difference_words,
    restore_difference_words,
)


def controlled_greater_program(
    dtype: RegisterType, *, control_value: int = 1, adjoint: bool = False
) -> ProgramRecord:
    """XOR a>b into a dirty target for either source control polarity.

    Signed inputs receive sign extension. Unsigned and Montgomery inputs receive
    zero prefixes. The exact source adder inverse owns measurement feedback and
    frees its computed sum, then the input prefixes are cleared and freed.
    The adjoint reverses the logical calls literally. The source signed inverse
    has incompatible directional port names and remains explicitly rejected.
    """
    if type(adjoint) is not bool:
        raise ValueError("Controlled comparison direction requires a boolean")
    if adjoint and dtype.encoding == "signed":
        raise ValueError(
            "Source signed comparator adjoint has incompatible sign-extension port names"
        )
    width = comparison_width(dtype)
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Comparison control value must be binary")
    bit = quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=bit, a=dtype, b=dtype, target=bit)
    )
    a, b = builder.inputs["a"], builder.inputs["b"]
    assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    a, b = prepare_difference_words(builder, dtype, a, b)
    result = builder.call_program(
        out_of_place_addition_program(width + 1), {"a": a, "b": b}
    )
    complement = bitwise_word_program(unsigned_register_type(width + 2), "bitwise_not")
    c = builder.call_program(complement, {"x": result["c"]})["x"]
    assert isinstance(c, RegisterBinding)
    cs = builder.split(c)
    copied = builder.call_program(
        multi_control_x_program((control_value, 1)),
        {
            "controls": np.array([builder.inputs["ctrl"], cs[1]], dtype=object),
            "target": builder.inputs["target"],
        },
    )
    ctrl, cs[1] = np.asarray(copied["controls"])
    c = builder.join(cs)
    c = builder.call_program(complement, {"x": c})["x"]
    restored = builder.call_program(
        out_of_place_addition_program(width + 1, adjoint=True),
        {"a": result["a"], "b": result["b"], "c": c},
    )
    a, b = restored["a"], restored["b"]
    assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    a, b = restore_difference_words(builder, dtype, a, b)
    record = builder.finalize(
        {"ctrl": ctrl, "a": a, "b": b, "target": copied["target"]}
    )
    if not adjoint:
        return record
    compute = out_of_place_addition_program(width + 1)
    erase = out_of_place_addition_program(width + 1, adjoint=True)
    copy = multi_control_x_program((control_value, 1))
    wide = replace(dtype, quantum_bits=width + 1)
    complement_input = bitwise_word_program(wide, "bitwise_not")
    pairs = [
        (compute, erase),
        (erase, compute),
        (copy, copy),
        (complement, complement),
        (complement_input, complement_input),
    ]
    return adjoint_component_calls(record, tuple(pairs))
