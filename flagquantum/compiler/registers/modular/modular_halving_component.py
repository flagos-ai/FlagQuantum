# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModDbl.adjoint().
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Compose the exact doubling adjoint and restore its original word roles."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.constant_addition_component import constant_addition_program
from ..comparison._component_gates import comparison_bit_functions
from ..component_builder import ComponentProgramBuilder
from .modular_doubling_component import modular_doubling_program


def modular_halving_program(dtype: RegisterType, modulus: int) -> ProgramRecord:
    """Consume a valid doubled residue and return the original MSB word.

    The input low bit becomes clean work. The allocated high bit becomes data.
    Specialized measured addition adjoints consume their exact forward images.
    """
    modular_doubling_program(dtype, modulus)
    width = dtype.quantum_bits
    assert isinstance(width, int)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype), comparison_bit_functions()
    )
    sign, high = builder.allocate(), builder.allocate()
    x = builder.inputs["x"]
    assert isinstance(x, RegisterBinding)
    pieces = builder.split(x)
    lower = builder.call_dict("comparison_x", {"q": pieces[-1]})["q"]
    parity = builder.call_dict("comparison_cx", {"ctrl": lower, "target": sign})
    lower = builder.call_dict("comparison_x", {"q": parity["ctrl"]})["q"]
    corrected = builder.call_program(
        constant_addition_program(
            unsigned_register_type(width + 1), modulus, control_value=1, adjoint=True
        ),
        {
            "ctrl": parity["target"],
            "x": builder.join(
                [high, *pieces[:-1], lower], unsigned_register_type(width + 1)
            ),
        },
    )
    x = corrected["x"]
    assert isinstance(x, RegisterBinding)
    wide = builder.join(
        [corrected["ctrl"], *builder.split(x)], signed_register_type(width + 2)
    )
    restored = builder.call_program(
        constant_addition_program(
            signed_register_type(width + 2),
            (-modulus) % (1 << (width + 2)),
            adjoint=True,
        ),
        {"x": wide},
    )["x"]
    assert isinstance(restored, RegisterBinding)
    pieces = builder.split(restored)
    builder.free(pieces[0])
    builder.free(pieces[-1])
    return builder.finalize({"x": builder.join(pieces[1:-1], dtype)})
