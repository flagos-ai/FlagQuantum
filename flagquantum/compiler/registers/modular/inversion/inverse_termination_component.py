# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep1.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Compose zero detection and retained termination history through shared calls."""

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
from ...control.conjunction_program import conjunction_program


def inverse_termination_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Preserve v/m/f/is_terminal and erase zero-test work by measurement.

    v is MSB-first. History flags remain logical data in either direction.
    The one-bit source MultiAnd failure is preserved as an explicit refusal.
    """
    word = montgomery_register_type(width)
    if width < 2 or type(adjoint) is not bool:
        raise ValueError("Termination requires at least two bits and boolean direction")
    bit = quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(v=word, m=bit, f=bit, is_terminal=bit),
        comparison_bit_functions(),
    )
    values = dict(builder.inputs)
    if adjoint:
        for target in ("is_terminal", "f"):
            result = builder.call_dict(
                "comparison_cx", {"ctrl": values["m"], "target": values[target]}
            )
            values["m"], values[target] = result["ctrl"], result["target"]
    v = values["v"]
    assert isinstance(v, RegisterBinding)
    pieces = builder.split(v)
    pattern = (0,) * width + (1,)
    computed = builder.call_program(
        conjunction_program(pattern, name="main"),
        {"ctrl": np.array([*pieces, values["f"]], dtype=object)},
    )
    copied = builder.call_dict(
        "comparison_cx", {"ctrl": computed["target"], "target": values["m"]}
    )
    computed["target"], values["m"] = copied["ctrl"], copied["target"]
    restored = builder.call_program(
        conjunction_program(pattern, name="main", uncompute=True), computed
    )
    controls = np.asarray(restored["ctrl"])
    values["v"], values["f"] = builder.join(controls[:-1], word), controls[-1]
    if not adjoint:
        for target in ("f", "is_terminal"):
            result = builder.call_dict(
                "comparison_cx", {"ctrl": values["m"], "target": values[target]}
            )
            values["m"], values[target] = result["ctrl"], result["target"]
    return builder.finalize(values)
