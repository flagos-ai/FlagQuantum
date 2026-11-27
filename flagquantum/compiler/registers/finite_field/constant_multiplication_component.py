# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2MulK at
# bcb17bb25360ed68ce3bc084645db3e929799389, gf2_multiplication.py.

"""Compose constant field scaling through the shared stored linear graph."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .field_multiplication import _constant_matrix
from .field_word import field_word_width
from .linear_component import binary_linear_decomposition_program, binary_linear_program


def constant_multiplication_program(
    dtype: RegisterType, constant: int, *, modulus: int, adjoint: bool = False
) -> ProgramRecord:
    """Scale MSB field word g by a nonzero constant, or invert it exactly.

    The inverse uses the accepted exact PLU adjoint. The frozen source linear
    adjoint ignores its flag, so affected source circuits remain defect witnesses.
    This operation introduces zero work, measurement or output-routing gates.
    """
    width = field_word_width(dtype, modulus)
    if type(constant) is not int or not 0 < constant < 1 << width:
        raise ValueError("Field multiplier must be a nonzero fitting integer")
    matrix = tuple(tuple(row) for row in _constant_matrix(width, constant, modulus))
    linear = binary_linear_program(matrix, adjoint=adjoint)
    return _constant_word_program(dtype, linear)


def constant_multiplication_decomposition_program(
    dtype: RegisterType, constant: int, *, modulus: int, adjoint: bool = False
) -> ProgramRecord:
    """Preserve zero-constant and ignored-adjoint source physical synthesis.

    The literal reversible circuit need not equal numerical multiplication.
    The mathematical constant entry retains its nonzero and exact-inverse rules.
    """
    width = field_word_width(dtype, modulus)
    if type(constant) not in (int, bool) or not 0 <= constant < 1 << width:
        raise ValueError("Literal field multiplier must be a fitting integer")
    matrix = tuple(tuple(row) for row in _constant_matrix(width, constant, modulus))
    linear = binary_linear_decomposition_program(matrix, adjoint=adjoint)
    return _constant_word_program(dtype, linear)


def _constant_word_program(dtype: RegisterType, linear: ProgramRecord) -> ProgramRecord:
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(g=dtype))
    word = builder.inputs["g"]
    assert isinstance(word, RegisterBinding)
    output = builder.call_program(
        linear, {"q": builder.split(word)[::-1]}, entry=linear.functions[0].name
    )["q"]
    return builder.finalize(
        {"g": builder.join(np.asarray(output, dtype=object)[::-1], dtype)}
    )
