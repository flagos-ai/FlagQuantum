# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls from Qualtran v0.7.0 cryptography/ecc/ec_phase_estimate_r.py.

"""Thread source phase controls through point additions and the terminal shim."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def curve_phase_estimation_program(
    signature: Signature,
    *,
    plus: ProgramRecord,
    additions: tuple[ProgramRecord, ...],
    measurement: ProgramRecord,
) -> ProgramRecord:
    """Retain scalar versus shaped controls and their original terminal order.

    The root owns the newly prepared phase bits until measurement consumes them.
    Coordinate words remain caller-owned. The original terminal declaration has
    no classical return; this composition does not invent one or replace it with
    the separate executable readout operation.
    """
    width = signature.get_left("x").bitsize
    if type(width) is not int or width < 1 or not additions:
        raise ValueError("Curve phase composition requires concrete nonempty controls")
    builder = ComponentProgramBuilder(signature)
    prepare = builder.import_program(plus, entry=plus.functions[0].name)
    ctrl = np.array(
        [builder.call_dict(prepare, {})["q"] for _ in range(width)], dtype=object
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    chunks = np.split(ctrl, len(additions))
    for index, child in enumerate(additions):
        port = Signature(child.functions[0].registers or ()).get_left("ctrl")
        result = builder.call_program(
            child,
            {
                "ctrl": chunks[index] if port.shape_symbolic else chunks[index][0],
                "x": x,
                "y": y,
            },
            entry=child.functions[0].name,
        )
        chunks[index] = np.asarray(result["ctrl"]).reshape(chunks[index].shape)
        x, y = result["x"], result["y"]
    builder.call_program(
        measurement, {"x": np.concatenate(chunks)}, entry=measurement.functions[0].name
    )
    return builder.finalize({"x": x, "y": y})
