# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal call order from locked Qualtran v0.7.0 first_quantization/prepare_nu.py.

"""Compose momentum-transfer preparation without assigning opaque leaf bodies."""

from collections.abc import Mapping

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def nu_preparation_program(
    signature: Signature,
    leaves: Mapping[str, ProgramRecord],
    uniform: ProgramRecord,
) -> ProgramRecord:
    """Preserve the six source calls and producer/consumer ownership of flags."""
    if leaves.keys() != {"mu", "nu", "zero", "less", "inequality"}:
        raise ValueError("Momentum preparation requires its five source leaves")
    builder = ComponentProgramBuilder(signature)
    children = {
        role: builder.import_program(record, entry=record.functions[0].name)
        for role, record in leaves.items()
    }
    uniform_name = builder.import_program(uniform, entry=uniform.functions[0].name)
    state = dict(builder.inputs)
    mu = builder.call_dict(children["mu"], {"mu": state["mu"]})
    transfer = builder.call_dict(children["nu"], {"mu": mu["mu"], "nu": state["nu"]})
    zero = builder.call_dict(children["zero"], {"nu": transfer["nu"]})
    less = builder.call_dict(children["less"], {"mu": transfer["mu"], "nu": zero["nu"]})
    sample = builder.call_dict(uniform_name, {"target": state["m"]})
    result = builder.call_dict(
        children["inequality"],
        {
            "mu": less["mu"],
            "nu": less["nu"],
            "m": sample["target"],
            "flag_mu_prep": mu["flag"],
            "flag_minus_zero": zero["flag_minus_zero"],
            "flag_ineq": less["flag_nu_lt_mu"],
            "succ": state["flag_nu"],
        },
    )
    return builder.finalize(
        {
            "mu": result["mu"],
            "nu": result["nu"],
            "m": result["m"],
            "flag_nu": result["succ"],
        }
    )
