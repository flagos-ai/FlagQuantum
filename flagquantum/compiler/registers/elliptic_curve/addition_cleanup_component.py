# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_add.py, _ECAddStepFive.

"""Consume the promised slope and restore the source coordinate roles."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def addition_cleanup_program(
    signature: Signature,
    *,
    inverse: ProgramRecord,
    multiply: ProgramRecord,
    select: ProgramRecord,
    clear_slope: ProgramRecord,
    multiply_adjoint: ProgramRecord,
    inverse_adjoint: ProgramRecord,
    controlled_negate: ProgramRecord,
    add: ProgramRecord,
    controlled_subtract: ProgramRecord,
) -> ProgramRecord:
    """Preserve the original LEFT slope promise and exact retained histories."""
    builder = ComponentProgramBuilder(signature)
    invert, mul, choose, clear, mul_adj, invert_adj, negate, plus, sub = (
        builder.import_program(child, entry=child.functions[0].name)
        for child in (
            inverse,
            multiply,
            select,
            clear_slope,
            multiply_adjoint,
            inverse_adjoint,
            controlled_negate,
            add,
            controlled_subtract,
        )
    )
    ctrl, a, b, x, y, lam_r, lam = (
        builder.inputs[name] for name in ("ctrl", "a", "b", "x", "y", "lam_r", "lam")
    )
    x, junk = builder.call_tuple(invert, {"x": x})
    x, y, z4, z3, reduced = builder.call_tuple(mul, {"x": x, "y": y})
    assert isinstance(z4, RegisterBinding) and isinstance(lam, RegisterBinding)
    z4_bits, lam_bits = builder.split(z4), builder.split(lam)
    for i in range(len(z4_bits)):
        controls, lam_bits[i] = builder.call_tuple(
            choose, {"controls": np.array([ctrl, z4_bits[i]]), "target": lam_bits[i]}
        )
        assert isinstance(controls, np.ndarray)
        ctrl, z4_bits[i] = controls
    z4 = builder.join(z4_bits, signature.get_left("lam").dtype)
    lam = builder.join(lam_bits, signature.get_left("lam").dtype)
    ctrl, x, lam_r, lam = builder.call_tuple(
        clear, {"ctrl1": ctrl, "ctrl2": x, "x": lam_r, "y": lam}
    )
    assert isinstance(lam, RegisterBinding)
    builder.free(lam)
    x, y = builder.call_tuple(
        mul_adj, {"x": x, "y": y, "target": z4, "qrom_indices": z3, "reduced": reduced}
    )
    x = builder.call_dict(invert_adj, {"x": x, "junk": junk})["x"]
    ctrl, x = builder.call_tuple(negate, {"ctrl": ctrl, "x": x})
    a, x = builder.call_tuple(plus, {"x": a, "y": x})
    ctrl, b, y = builder.call_tuple(sub, {"ctrl": ctrl, "x": b, "y": y})
    return builder.finalize(
        {"ctrl": ctrl, "a": a, "b": b, "x": x, "y": y, "lam_r": lam_r}
    )
