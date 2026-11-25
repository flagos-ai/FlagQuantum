# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CModMulK.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Compose controlled scale, inverse-scale clear and the exact exchange."""

from math import gcd

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..primitives.controlled_swap_component import controlled_swap_program
from .scaled_modular_addition_component import scaled_modular_addition_program


def modular_multiplication_program(
    dtype: RegisterType, modulus: int, multiplier: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Multiply a valid MSB-first residue under the source positive control.

    The source's result word is allocated until after both scale-add calls and
    the exact controlled swap. The inverse scale keeps its original negative
    representative. The source full-ring loader failure stays observable.
    The exact inverse retains both scales and reverses their typed calls.
    """
    if type(adjoint) is not bool:
        raise ValueError("Modular multiplication direction requires a boolean")
    width = executable_register_width(dtype)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError("Modular multiplication requires unsigned or Montgomery words")
    if type(modulus) is not int or not 1 < modulus <= 1 << width:
        raise ValueError("Modular multiplication requires a fitting positive modulus")
    if (
        type(multiplier) is not int
        or not 0 < multiplier < modulus
        or gcd(multiplier, modulus) != 1
    ):
        raise ValueError("Modular multiplier must be a positive coprime residue")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=quantum_bit_type(), x=dtype)
    )
    result = builder.allocate(width)
    added = builder.call_program(
        scaled_modular_addition_program(width, modulus, multiplier),
        {"ctrl": builder.inputs["ctrl"], "x": builder.inputs["x"], "y": result},
    )
    cleared = builder.call_program(
        scaled_modular_addition_program(width, modulus, -pow(multiplier, -1, modulus)),
        {"ctrl": added["ctrl"], "x": added["y"], "y": added["x"]},
    )
    swapped = builder.call_program(
        controlled_swap_program(width),
        {"ctrl": cleared["ctrl"], "x": cleared["y"], "y": cleared["x"]},
    )
    cleared_result = swapped["y"]
    assert isinstance(cleared_result, RegisterBinding)
    builder.free(cleared_result)
    record = builder.finalize({"ctrl": swapped["ctrl"], "x": swapped["x"]})
    if not adjoint:
        return record
    scale = scaled_modular_addition_program(width, modulus, multiplier)
    inverse_scale = -pow(multiplier, -1, modulus)
    clear = scaled_modular_addition_program(width, modulus, inverse_scale)
    swap = controlled_swap_program(width)
    return adjoint_component_calls(
        record,
        (
            (
                scale,
                scaled_modular_addition_program(
                    width, modulus, multiplier, adjoint=True
                ),
            ),
            (
                clear,
                scaled_modular_addition_program(
                    width, modulus, inverse_scale, adjoint=True
                ),
            ),
            (swap, swap),
        ),
    )
