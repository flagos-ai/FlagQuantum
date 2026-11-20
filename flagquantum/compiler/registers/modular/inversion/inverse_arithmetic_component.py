# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep5.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Compose the source complemented subtraction and coefficient addition."""

import numpy as np

from .....core.ir.program import ProgramRecord
from .....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from .....core.ir.program.signatures import Signature
from ...arithmetic.controlled_addition_component import controlled_addition_program
from ...bits.bitwise_component import bitwise_word_program
from ...component_builder import ComponentProgramBuilder
from ...control.conjunction_program import and_component_program


def inverse_arithmetic_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Update finite-width v and s under f AND NOT b and restore clean work.

    u/v/r/s are MSB-first Montgomery words. The unchanged source callback's
    unbounded subtraction/addition agrees only where its intermediate values fit.
    """
    word, bit = montgomery_register_type(width), quantum_bit_type()
    if type(adjoint) is not bool:
        raise ValueError("Inverse arithmetic direction requires a boolean")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(u=word, v=word, r=word, s=word, b=bit, f=bit)
    )
    values = dict(builder.inputs)
    computed = builder.call_program(
        and_component_program((1, 0), name="main"),
        {"ctrl": np.array([values["f"], values["b"]], dtype=object)},
    )
    addition = controlled_addition_program(word, adjoint=adjoint)
    complement = bitwise_word_program(word, "bitwise_not")
    pairs = (("u", "v"), ("r", "s"))
    for left, right in reversed(pairs) if adjoint else pairs:
        if right == "v":
            values[right] = builder.call_program(complement, {"x": values[right]})["x"]
        added = builder.call_program(
            addition,
            {"ctrl": computed["target"], "a": values[left], "b": values[right]},
        )
        computed["target"], values[left], values[right] = (
            added["ctrl"],
            added["a"],
            added["b"],
        )
        if right == "v":
            values[right] = builder.call_program(complement, {"x": values[right]})["x"]
    restored = builder.call_program(
        and_component_program((1, 0), name="main", uncompute=True),
        computed,
    )
    values["f"], values["b"] = np.asarray(restored["ctrl"])
    return builder.finalize(values)
