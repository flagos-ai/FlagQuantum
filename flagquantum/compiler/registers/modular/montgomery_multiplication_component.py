# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 DirtyOutOfPlaceMontgomeryModMul.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Own Montgomery product/history outputs around the inlined window traversal."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .windowed_multiplication_component import windowed_multiplication_program


def montgomery_multiplication_program(
    width: int, window: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Create or consume source product, index-history and reduction ports.

    MSB-first x/y are through words. Forward RIGHT outputs transfer owned work
    to the caller. Inverse LEFT inputs require caller-owned matching source
    outputs and are released after the specialized window inverse. Generic
    exterior-value promises and source callback defects remain explicit.
    """
    core = windowed_multiplication_program(width, window, modulus, adjoint=adjoint)
    dtype = montgomery_register_type(width, modulus)
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", dtype),
                Register("y", dtype),
                Register("target", dtype, side=side),
                Register("qrom_indices", dtype, side=side),
                Register("reduced", quantum_bit_type(), side=side),
            )
        )
    )
    values = dict(builder.inputs)
    if not adjoint:
        values["target"] = builder.allocate(width, montgomery_register_type(width))
        values["qrom_indices"] = builder.allocate(
            width, montgomery_register_type(width)
        )
        values["reduced"] = builder.allocate(1)
    result = builder.inline_calls(core, values)
    if adjoint:
        for name in ("reduced", "qrom_indices", "target"):
            value = result.pop(name)
            assert isinstance(value, RegisterBinding)
            builder.free(value)
    return builder.finalize(result)
