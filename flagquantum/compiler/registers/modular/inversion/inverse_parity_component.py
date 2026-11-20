# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep2.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Compose sequential parity predicates and branch updates with owned work."""

import numpy as np

from .....core.ir.program import ProgramRecord
from .....core.ir.program.register_bindings import RegisterBinding
from .....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from .....core.ir.program.signatures import Signature
from ...comparison._component_gates import comparison_bit_functions
from ...component_builder import ComponentProgramBuilder
from ...control.conjunction_program import and_component_program, conjunction_program


def inverse_parity_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Retain u/v/b/a/m/f and consume each predicate's exact measured image.

    Words are MSB-first. The second predicate reads the updated a flag forward.
    The inverse restores m before a and uses the same clean conjunction channels.
    """
    word, bit = montgomery_register_type(width), quantum_bit_type()
    if type(adjoint) is not bool:
        raise ValueError("Parity direction requires a boolean")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(u=word, v=word, b=bit, a=bit, m=bit, f=bit),
        comparison_bit_functions(),
    )
    values = dict(builder.inputs)
    if adjoint:
        for control in ("m", "a"):
            copied = builder.call_dict(
                "comparison_cx", {"ctrl": values[control], "target": values["b"]}
            )
            values[control], values["b"] = copied["ctrl"], copied["target"]
    u, v = values["u"], values["v"]
    assert isinstance(u, RegisterBinding) and isinstance(v, RegisterBinding)
    us, vs = builder.split(u), builder.split(v)
    stages = ("m", "a") if adjoint else ("a", "m")
    for target in stages:
        first = target == "a"
        compute = (
            and_component_program((1, 0), name="main")
            if first
            else conjunction_program((1, 0, 0), name="main")
        )
        erase = (
            and_component_program((1, 0), name="main", uncompute=True)
            if first
            else conjunction_program((1, 0, 0), name="main", uncompute=True)
        )
        controls = (
            [values["f"], us[-1]] if first else [values["f"], vs[-1], values["a"]]
        )
        computed = builder.call_program(
            compute, {"ctrl": np.array(controls, dtype=object)}
        )
        copied = builder.call_dict(
            "comparison_cx", {"ctrl": computed["target"], "target": values[target]}
        )
        computed["target"], values[target] = copied["ctrl"], copied["target"]
        restored = np.asarray(builder.call_program(erase, computed)["ctrl"])
        values["f"] = restored[0]
        if first:
            us[-1] = restored[1]
        else:
            vs[-1], values["a"] = restored[1], restored[2]
    if not adjoint:
        for control in ("a", "m"):
            copied = builder.call_dict(
                "comparison_cx", {"ctrl": values[control], "target": values["b"]}
            )
            values[control], values["b"] = copied["ctrl"], copied["target"]
    values["u"], values["v"] = builder.join(us, word), builder.join(vs, word)
    return builder.finalize(values)
