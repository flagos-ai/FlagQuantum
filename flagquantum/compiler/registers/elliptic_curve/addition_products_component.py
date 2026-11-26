# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_add.py, _ECAddStepFour.

"""Thread the two original products and consume their exact histories."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ._word_copy import copy_word


def addition_products_program(
    signature: Signature,
    *,
    zero: ProgramRecord,
    cnot: ProgramRecord,
    multiply: ProgramRecord,
    multiply_adjoint: ProgramRecord,
    subtract: ProgramRecord,
) -> ProgramRecord:
    """Retain x/y/lam and privately erase the copied slope and both products."""
    builder = ComponentProgramBuilder(signature)
    initialize, copy, mul, mul_adj, sub = (
        builder.import_program(child, entry=child.functions[0].name)
        for child in (zero, cnot, multiply, multiply_adjoint, subtract)
    )
    x, y, lam = (builder.inputs[name] for name in ("x", "y", "lam"))
    dtype = signature.get_left("lam").dtype
    z4 = builder.call_dict(initialize, {})["val"]
    lam, z4 = copy_word(builder, lam, z4, dtype, copy)
    z4, lam, z3, z2, reduced = builder.call_tuple(mul, {"x": z4, "y": lam})
    z3, x = builder.call_tuple(sub, {"x": z3, "y": x})
    z4, lam = builder.call_tuple(
        mul_adj,
        {"x": z4, "y": lam, "target": z3, "qrom_indices": z2, "reduced": reduced},
    )
    lam, z4 = copy_word(builder, lam, z4, dtype, copy)
    assert isinstance(z4, RegisterBinding)
    builder.free(z4)
    x, lam, z3, z4, reduced = builder.call_tuple(mul, {"x": x, "y": lam})
    z3, y = copy_word(builder, z3, y, dtype, copy)
    x, lam = builder.call_tuple(
        mul_adj,
        {"x": x, "y": lam, "target": z3, "qrom_indices": z4, "reduced": reduced},
    )
    return builder.finalize({"x": x, "y": y, "lam": lam})
