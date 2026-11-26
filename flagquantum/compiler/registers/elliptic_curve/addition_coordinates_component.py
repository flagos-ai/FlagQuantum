# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_add.py, _ECAddStepThree.

"""Compose the controlled coordinate differences and erase the tripling word."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ._word_copy import copy_word


def addition_coordinates_program(
    signature: Signature,
    *,
    multiply: ProgramRecord,
    multiply_adjoint: ProgramRecord,
    controlled_subtract: ProgramRecord,
    zero: ProgramRecord,
    cnot: ProgramRecord,
    double: ProgramRecord,
    add: ProgramRecord,
    controlled_add: ProgramRecord,
    add_adjoint: ProgramRecord,
    double_adjoint: ProgramRecord,
) -> ProgramRecord:
    """Keep source word aliases and privately own the product and 3a work."""
    builder = ComponentProgramBuilder(signature)
    mul, mul_adj, csub, initialize, copy, dbl, plus, cplus, plus_adj, dbl_adj = (
        builder.import_program(child, entry=child.functions[0].name)
        for child in (
            multiply,
            multiply_adjoint,
            controlled_subtract,
            zero,
            cnot,
            double,
            add,
            controlled_add,
            add_adjoint,
            double_adjoint,
        )
    )
    ctrl, a, b, x, y, lam = (
        builder.inputs[name] for name in ("ctrl", "a", "b", "x", "y", "lam")
    )
    dtype = signature.get_left("a").dtype
    x, lam, z1, z2, reduced = builder.call_tuple(mul, {"x": x, "y": lam})
    ctrl, z1, y = builder.call_tuple(csub, {"ctrl": ctrl, "x": z1, "y": y})
    x, lam = builder.call_tuple(
        mul_adj,
        {"x": x, "y": lam, "target": z1, "qrom_indices": z2, "reduced": reduced},
    )
    z1 = builder.call_dict(initialize, {})["val"]
    a, z1 = copy_word(builder, a, z1, dtype, copy)
    z1 = builder.call_dict(dbl, {"x": z1})["x"]
    a, z1 = builder.call_tuple(plus, {"x": a, "y": z1})
    ctrl, z1, x = builder.call_tuple(cplus, {"ctrl": ctrl, "x": z1, "y": x})
    a, z1 = builder.call_tuple(plus_adj, {"x": a, "y": z1})
    z1 = builder.call_dict(dbl_adj, {"x": z1})["x"]
    a, z1 = copy_word(builder, a, z1, dtype, copy)
    assert isinstance(z1, RegisterBinding)
    builder.free(z1)
    return builder.finalize({"ctrl": ctrl, "a": a, "b": b, "x": x, "y": y, "lam": lam})
