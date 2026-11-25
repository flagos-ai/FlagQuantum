# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_addition.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py

"""Store controlled modular addition with original casts and correction stages."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.constant_addition_component import constant_addition_program
from ..arithmetic.controlled_addition_component import controlled_addition_program
from ..comparison._component_gates import comparison_bit_functions
from ..comparison.controlled_component import controlled_greater_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..layout.component_cast import cast_word_function
from .controlled_modular_addition import controlled_modular_requirements


def controlled_modular_addition_program(
    dtype: RegisterType, modulus: int, *, control_value: int = 1, adjoint: bool = False
) -> ProgramRecord:
    """Retain ctrl/x and conditionally add x modulo modulus into y.

    MSB-first uint or Montgomery words retain their external descriptor. Both
    control polarities follow the original CAdd and comparator. Corrections run
    in both control sectors. Sign and arithmetic work start clean and are
    released under the original residue-domain promise, without input repair.
    The adjoint reverses the same logical callees before measurement lowering.
    """
    if type(adjoint) is not bool:
        raise ValueError("Controlled modular addition direction requires a boolean")
    width = executable_register_width(dtype)
    controlled_modular_requirements(width)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError(
            "Controlled modular addition requires unsigned or Montgomery words"
        )
    if type(modulus) is not int or not 1 <= modulus <= 1 << width:
        raise ValueError("Modulus must be a positive integer fitting the operand")
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Controlled modular control value must be zero or one")
    word, wide = unsigned_register_type(width), unsigned_register_type(width + 1)
    addition = controlled_addition_program(word, wide, control_value=control_value)
    subtraction = constant_addition_program(wide, (-modulus) % (1 << (width + 1)))
    correction = constant_addition_program(word, modulus, control_value=1)
    comparison = controlled_greater_program(word, control_value=control_value)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=quantum_bit_type(), x=dtype, y=dtype),
        (
            *comparison_bit_functions(),
            cast_word_function("mod_cast_unsigned", dtype, word),
            cast_word_function("mod_cast_return", word, dtype),
        ),
    )
    y = builder.inputs["y"]
    assert isinstance(y, RegisterBinding)
    ys = builder.split(y)
    sign = builder.allocate()
    x = builder.call_dict("mod_cast_unsigned", {"reg": builder.inputs["x"]})["reg"]
    y = builder.join([sign, *ys], wide)
    added = builder.call_program(
        addition,
        {"ctrl": builder.inputs["ctrl"], "a": x, "b": y},
    )
    y = builder.call_program(
        subtraction,
        {"x": added["b"]},
    )["x"]
    assert isinstance(y, RegisterBinding)
    ys = builder.split(y)
    sign, y = ys[0], builder.join(ys[1:])
    corrected = builder.call_program(
        correction,
        {"ctrl": sign, "x": y},
    )
    compared = builder.call_program(
        comparison,
        {
            "ctrl": added["ctrl"],
            "a": added["a"],
            "b": corrected["x"],
            "target": corrected["ctrl"],
        },
    )
    cleared_sign = builder.call_dict("comparison_x", {"q": compared["target"]})["q"]
    assert isinstance(cleared_sign, RegisterBinding)
    builder.free(cleared_sign)
    x = builder.call_dict("mod_cast_return", {"reg": compared["a"]})["reg"]
    y = builder.call_dict("mod_cast_return", {"reg": compared["b"]})["reg"]
    record = builder.finalize({"ctrl": compared["ctrl"], "x": x, "y": y})
    if not adjoint:
        return record
    x_gate = ProgramRecord((comparison_bit_functions()[0],))
    return adjoint_component_calls(
        record,
        (
            (
                addition,
                controlled_addition_program(
                    word, wide, control_value=control_value, adjoint=True
                ),
            ),
            (
                subtraction,
                constant_addition_program(
                    wide, (-modulus) % (1 << (width + 1)), adjoint=True
                ),
            ),
            (
                correction,
                constant_addition_program(word, modulus, control_value=1, adjoint=True),
            ),
            (
                comparison,
                controlled_greater_program(
                    word, control_value=control_value, adjoint=True
                ),
            ),
            (x_gate, x_gate),
        ),
    )
