# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_addition.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py

"""Store the widened Montgomery addition and original sign cleanup graph."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import montgomery_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.addition_component import addition_component_program
from ..arithmetic.constant_addition_component import constant_addition_program
from ..comparison._component_gates import comparison_bit_functions
from ..comparison.linear_component import linear_greater_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from .modular_addition import modular_add_requirements


def modular_addition_program(
    width: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Preserve x and add it modulo modulus into y using the source graph.

    Both MSB-first ports retain the declared Montgomery modulus. The internal
    Montgomery views omit that modulus exactly as upstream. Clean junk/sign
    prefixes and arithmetic work are owned until release. Physical cleanup is
    promised for x in [0, modulus] and y in [0, modulus), subject to source
    defects recorded in the file audit.
    The adjoint reverses the same typed children using their exact inverses.
    """
    if type(adjoint) is not bool:
        raise ValueError("Modular addition direction must be Boolean")
    modular_add_requirements(width)
    if type(modulus) is not int or not 1 <= modulus <= 1 << width:
        raise ValueError("Modulus must be a positive integer fitting the operand")
    dtype = montgomery_register_type(width, modulus)
    narrow, wide = montgomery_register_type(width), montgomery_register_type(width + 1)
    record = modular_addition_component_program(
        dtype,
        constant_addition_program(wide, (-modulus) % (1 << (width + 1))),
        constant_addition_program(narrow, modulus, control_value=1),
    )
    if not adjoint:
        return record
    x_gate = ProgramRecord((comparison_bit_functions()[0],))
    return adjoint_component_calls(
        record,
        (
            (
                addition_component_program(
                    width + 1, left_dtype=wide, right_dtype=wide
                ),
                addition_component_program(
                    width + 1, left_dtype=wide, right_dtype=wide, adjoint=True
                ),
            ),
            (
                constant_addition_program(wide, (-modulus) % (1 << (width + 1))),
                constant_addition_program(
                    wide, (-modulus) % (1 << (width + 1)), adjoint=True
                ),
            ),
            (
                constant_addition_program(narrow, modulus, control_value=1),
                constant_addition_program(
                    narrow, modulus, control_value=1, adjoint=True
                ),
            ),
            (
                linear_greater_program(width),
                linear_greater_program(width, adjoint=True),
            ),
            (x_gate, x_gate),
        ),
    )


def modular_addition_component_program(
    dtype: RegisterType, subtraction: ProgramRecord, correction: ProgramRecord
) -> ProgramRecord:
    """Retain the source widening and sign cleanup around explicit constants.

    The caller owns the actual minus-modulus and controlled plus-modulus
    children. Unresolved constants may remain true source declarations;
    this template supplies neither numerical encodings nor inferred prices.
    """
    width = executable_register_width(dtype)
    if dtype.encoding != "montgomery":
        raise ValueError("Modular addition requires Montgomery source words")
    modular_add_requirements(width)
    narrow, wide = montgomery_register_type(width), montgomery_register_type(width + 1)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype, y=dtype), comparison_bit_functions()
    )
    junk, sign = builder.allocate(), builder.allocate()
    x, y = builder.inputs["x"], builder.inputs["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    xs, ys = builder.split(x), builder.split(y)
    x = builder.join([junk, *xs], wide)
    y = builder.join([sign, *ys], wide)
    added = builder.call_program(
        addition_component_program(width + 1, left_dtype=wide, right_dtype=wide),
        {"a": x, "b": y},
    )
    added_x = added["a"]
    assert isinstance(added_x, RegisterBinding)
    xs = builder.split(added_x)
    junk, x = xs[0], builder.join(xs[1:], narrow)
    y = builder.call_program(
        subtraction,
        {"x": added["b"]},
        entry=subtraction.functions[0].name,
    )["x"]
    assert isinstance(y, RegisterBinding)
    ys = builder.split(y)
    sign, y = ys[0], builder.join(ys[1:], narrow)
    corrected = builder.call_program(
        correction,
        {"ctrl": sign, "x": y},
        entry=correction.functions[0].name,
    )
    compared = builder.call_program(
        linear_greater_program(width),
        {"a": x, "b": corrected["x"], "target": corrected["ctrl"]},
    )
    cleared_sign = builder.call_dict("comparison_x", {"q": compared["target"]})["q"]
    assert isinstance(cleared_sign, RegisterBinding)
    builder.free(junk)
    builder.free(cleared_sign)
    return builder.finalize({"x": compared["a"], "y": compared["b"]})
