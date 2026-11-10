# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GF2MulMBUC.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py

"""Build the owned binary-field product erasure graph with shared components."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_classical import discard_word_function, parity_word_function
from ..component_feedback import classical_cz_word_function
from ..component_measurement import measure_x_word_function
from ..component_rebindings import join_word_function, split_word_function
from .field_multiplication import _remainder, _validate_field


def product_erasure_program(dtype: RegisterType) -> ProgramRecord:
    """Consume a known field product, preserving its two input words.

    The descriptor must carry a concrete binary polynomial identity. Words use
    MSB-first physical ordering. Internal polynomial coefficients are reversed
    exactly as in the source. Destructive measurement retains physical projected
    outcomes, while classical owners are consumed by Discard. Caller-owned
    physical clearing remains a separate execution profile.
    """
    width = executable_register_width(dtype)
    parameters = dict(dtype.parameters)
    modulus = parameters.get("polynomial")
    if (
        dtype.encoding != "finite_field"
        or type(modulus) is not int
        or parameters.get("polynomial_field_characteristic") != 2
        or parameters.get("polynomial_field_degree") != 1
    ):
        raise ValueError("product erasure requires a resolved binary polynomial field")
    _validate_field(width, modulus)
    masks = tuple(_remainder(1 << (width + i), modulus) for i in range(width))
    functions = [
        split_word_function("split", dtype),
        join_word_function("join", dtype),
        measure_x_word_function("measure_x"),
        classical_cz_word_function("controlled_cz"),
        discard_word_function("discard"),
        *(
            parity_word_function(f"parity_{n}", n)
            for n in sorted({mask.bit_count() for mask in masks})
        ),
    ]
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("x", dtype),
                Register("y", dtype),
                Register("result", dtype, side=RegisterSide.LEFT),
            ]
        ),
        functions,
    )
    x, y, result = (
        np.asarray(
            builder.call_dict("split", {"reg": builder.inputs[name]})["reg"],
            dtype=object,
        )[::-1].copy()
        for name in ("x", "y", "result")
    )
    result = np.array(
        [builder.call_dict("measure_x", {"q": q})["c"] for q in result], dtype=object
    )
    for i in range(width):
        for j in range(i + 1):
            output = builder.call_dict(
                "controlled_cz", {"ctrl": result[i], "q1": x[j], "q2": y[i - j]}
            )
            result[i], x[j], y[i - j] = output["ctrl"], output["q1"], output["q2"]
    for i, mask in enumerate(masks):
        indices = [k for k in range(width) if mask >> k & 1]
        output = builder.call_dict(f"parity_{len(indices)}", {"x": result[indices]})
        result[indices], parity = output["x"], output["parity"]
        for j in range(i + 1, width):
            output = builder.call_dict(
                "controlled_cz", {"ctrl": parity, "q1": x[width - j + i], "q2": y[j]}
            )
            parity, x[width - j + i], y[j] = output["ctrl"], output["q1"], output["q2"]
        builder.call_dict("discard", {"c": parity})
    for bit in result:
        builder.call_dict("discard", {"c": bit})
    return builder.finalize(
        {
            name: builder.call_dict("join", {"reg": value[::-1]})["reg"]
            for name, value in (("x", x), ("y", y))
        }
    )
