# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep3.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Compose the half-comparison predicate and erase all owned difference bits."""

import numpy as np

from .....core.ir.program import ProgramRecord
from .....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from .....core.ir.program.signatures import Signature
from ...comparison._component_gates import comparison_bit_functions
from ...comparison.half_component import half_greater_program
from ...component_builder import ComponentProgramBuilder
from ...control.conjunction_program import conjunction_program


def inverse_ordering_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Retain u/v/b/a/m/f and restore comparison and conjunction workspace.

    The MSB-first difference and greater-than flag are owned child outputs.
    Both directions consume them through the measured comparison adjoint.
    """
    word, bit = montgomery_register_type(width), quantum_bit_type()
    if type(adjoint) is not bool:
        raise ValueError("Ordering direction requires a boolean")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(u=word, v=word, b=bit, a=bit, m=bit, f=bit),
        comparison_bit_functions(),
    )
    values = dict(builder.inputs)
    compared = builder.call_program(
        half_greater_program(word), {"a": values["u"], "b": values["v"]}
    )
    computed = builder.call_program(
        conjunction_program((1, 1, 0), name="main"),
        {
            "ctrl": np.array(
                [compared["target"], values["f"], values["b"]], dtype=object
            )
        },
    )
    for target in (("m", "a") if adjoint else ("a", "m")):
        copied = builder.call_dict(
            "comparison_cx", {"ctrl": computed["target"], "target": values[target]}
        )
        computed["target"], values[target] = copied["ctrl"], copied["target"]
    restored = np.asarray(
        builder.call_program(
            conjunction_program((1, 1, 0), name="main", uncompute=True), computed
        )["ctrl"]
    )
    compared["target"], values["f"], values["b"] = restored
    cleared = builder.call_program(half_greater_program(word, adjoint=True), compared)
    values["u"], values["v"] = cleared["a"], cleared["b"]
    return builder.finalize(values)
