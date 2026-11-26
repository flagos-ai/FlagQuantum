# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_add.py, _ECAddStepOne.

"""Compose the original elliptic-curve predicates with caller-owned flags."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def addition_flags_program(
    signature: Signature,
    zero: ProgramRecord,
    equals: ProgramRecord,
    negate: ProgramRecord,
    zero_pair: ProgramRecord,
    zero_flags: ProgramRecord,
) -> ProgramRecord:
    """Keep source calls and nominal word types; return five live logical bits."""
    builder = ComponentProgramBuilder(signature)
    zero_entry, equals_entry, negate_entry, pair_entry, flags_entry = (
        builder.import_program(child, entry=child.functions[0].name)
        for child in (zero, equals, negate, zero_pair, zero_flags)
    )
    a, b, x, y = (builder.inputs[name] for name in ("a", "b", "x", "y"))
    f1, f2, f3, f4, ctrl = (builder.call_dict(zero_entry, {})["q"] for _ in range(5))
    a, x, f1 = builder.call_tuple(equals_entry, {"x": a, "y": x, "target": f1})
    y = builder.call_dict(negate_entry, {"x": y})["x"]
    b, y, f2 = builder.call_tuple(equals_entry, {"x": b, "y": y, "target": f2})
    y = builder.call_dict(negate_entry, {"x": y})["x"]
    assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    ab_bits = np.concatenate((builder.split(a), builder.split(b)))
    ab, f3 = builder.call_tuple(pair_entry, {"controls": ab_bits, "target": f3})
    assert isinstance(ab, np.ndarray)
    a_bits, b_bits = np.split(ab, 2)
    a = builder.join(a_bits, signature.get_left("a").dtype)
    b = builder.join(b_bits, signature.get_left("b").dtype)
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    xy_bits = np.concatenate((builder.split(x), builder.split(y)))
    xy, f4 = builder.call_tuple(pair_entry, {"controls": xy_bits, "target": f4})
    assert isinstance(xy, np.ndarray)
    x_bits, y_bits = np.split(xy, 2)
    x = builder.join(x_bits, signature.get_left("x").dtype)
    y = builder.join(y_bits, signature.get_left("y").dtype)
    flags, ctrl = builder.call_tuple(
        flags_entry, {"controls": np.array([f2, f3, f4]), "target": ctrl}
    )
    assert isinstance(flags, np.ndarray)
    f2, f3, f4 = flags
    return builder.finalize(
        {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "ctrl": ctrl,
            "a": a,
            "b": b,
            "x": x,
            "y": y,
        }
    )
