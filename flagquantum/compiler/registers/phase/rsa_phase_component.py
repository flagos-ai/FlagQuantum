# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/rsa/rsa_phase_estimate.py.

"""Thread RSA initialization, controlled products and the original terminal shim."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def rsa_phase_estimation_program(
    width: int,
    plus: ProgramRecord,
    one: ProgramRecord,
    multipliers: tuple[ProgramRecord, ...],
    measurement: ProgramRecord,
) -> ProgramRecord:
    """Compose source-ordered calls without claiming terminal shim execution."""
    if type(width) is not int or width < 1 or len(multipliers) != 2 * width:
        raise ValueError("RSA composition requires a positive width and 2n multipliers")
    builder = ComponentProgramBuilder(Signature(()))
    plus_entry = builder.import_program(plus, entry=plus.functions[0].name)
    one_entry = builder.import_program(one, entry=one.functions[0].name)
    multiply_entries = tuple(
        builder.import_program(child, entry=child.functions[0].name)
        for child in multipliers
    )
    measurement_entry = builder.import_program(
        measurement, entry=measurement.functions[0].name
    )
    exponent = np.array(
        [builder.call_dict(plus_entry, {})["q"] for _ in range(2 * width)], dtype=object
    )
    x = builder.call_dict(one_entry, {})["val"]
    for entry, position in zip(
        multiply_entries, reversed(range(2 * width)), strict=True
    ):
        result = builder.call_dict(entry, {"ctrl": exponent[position], "x": x})
        exponent[position], x = result["ctrl"], result["x"]
    builder.call_dict(measurement_entry, {"x": exponent})
    assert isinstance(x, RegisterBinding)
    builder.free(x, dirty=True)
    return builder.finalize({})
