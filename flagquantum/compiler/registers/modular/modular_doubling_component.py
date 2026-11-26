# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModDbl.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Compose modular doubling with the source logical output/work exchange."""

from dataclasses import replace

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.constant_addition_component import constant_addition_program
from ..comparison._component_gates import comparison_bit_functions
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder


def modular_doubling_program(
    dtype: RegisterType, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Double an MSB-first residue and return the source's new data binding.

    The lower work bit becomes the final data bit. The old high data bit and
    sign are released after the original parity cleanup. Internal views retain
    the source signed subtract and unsigned controlled correction. Work starts
    clean and returns clean on the promised residue domain.
    The exact adjoint consumes the returned layout and reconstructs its predecessor.
    """
    if type(adjoint) is not bool:
        raise ValueError("Modular doubling direction must be Boolean")
    width = executable_register_width(dtype)
    if dtype.encoding not in {"uint", "montgomery"}:
        raise ValueError("Modular doubling requires unsigned or Montgomery words")
    if type(modulus) is not int or not 0 < modulus < 1 << width or modulus % 2 != 1:
        raise ValueError("Modular doubling requires a fitting positive odd modulus")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype), comparison_bit_functions()
    )
    lower, sign = builder.allocate(), builder.allocate()
    x = builder.inputs["x"]
    assert isinstance(x, RegisterBinding)
    wide = replace(dtype, quantum_bits=width + 2)
    joined = builder.join([sign, *builder.split(x), lower], wide)
    subtracted = builder.call_program(
        constant_addition_program(
            signed_register_type(width + 2), (-modulus) % (1 << (width + 2))
        ),
        {"x": joined},
    )["x"]
    assert isinstance(subtracted, RegisterBinding)
    pieces = builder.split(subtracted)
    corrected = builder.call_program(
        constant_addition_program(
            unsigned_register_type(width + 1), modulus, control_value=1
        ),
        {
            "ctrl": pieces[0],
            "x": builder.join(pieces[1:], replace(dtype, quantum_bits=width + 1)),
        },
    )
    x = corrected["x"]
    assert isinstance(x, RegisterBinding)
    pieces = builder.split(x)
    flipped_lower = builder.call_dict("comparison_x", {"q": pieces[-1]})["q"]
    parity = builder.call_dict(
        "comparison_cx", {"ctrl": flipped_lower, "target": corrected["ctrl"]}
    )
    restored_lower = builder.call_dict("comparison_x", {"q": parity["ctrl"]})["q"]
    output = builder.join([*pieces[1:-1], restored_lower], dtype)
    cleared_sign = parity["target"]
    assert isinstance(cleared_sign, RegisterBinding)
    builder.free(pieces[0])
    builder.free(cleared_sign)
    record = builder.finalize({"x": output})
    if not adjoint:
        return record
    x_gate, cx_gate = (
        ProgramRecord((function,)) for function in comparison_bit_functions()
    )
    return adjoint_component_calls(
        record,
        (
            (
                constant_addition_program(
                    signed_register_type(width + 2), (-modulus) % (1 << (width + 2))
                ),
                constant_addition_program(
                    signed_register_type(width + 2),
                    (-modulus) % (1 << (width + 2)),
                    adjoint=True,
                ),
            ),
            (
                constant_addition_program(
                    unsigned_register_type(width + 1), modulus, control_value=1
                ),
                constant_addition_program(
                    unsigned_register_type(width + 1),
                    modulus,
                    control_value=1,
                    adjoint=True,
                ),
            ),
            (x_gate, x_gate),
            (cx_gate, cx_gate),
        ),
    )
