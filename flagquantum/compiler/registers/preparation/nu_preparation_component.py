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
    *,
    adjoint: bool = False,
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
    if adjoint:
        result = builder.call_dict(
            children["inequality"],
            {
                "mu": state["mu"],
                "nu": state["nu"],
                "m": state["m"],
                "succ": state["flag_nu"],
            },
        )
        # The source greedy traversal gives symbolic leaves zero priority.
        # Reversing that traversal puts the independent uniform child here.
        sample = (
            builder.call_dict(uniform_name, {"target": result["m"]})
            if type(signature.get_left("mu").bitsize) is not int
            else None
        )
        less = builder.call_dict(
            children["less"],
            {
                "mu": result["mu"],
                "nu": result["nu"],
                "flag_nu_lt_mu": result["flag_ineq"],
            },
        )
        zero = builder.call_dict(
            children["zero"],
            {"nu": less["nu"], "flag_minus_zero": result["flag_minus_zero"]},
        )
        transfer = builder.call_dict(
            children["nu"], {"mu": less["mu"], "nu": zero["nu"]}
        )
        mu = builder.call_dict(
            children["mu"], {"mu": transfer["mu"], "flag": result["flag_mu_prep"]}
        )
        if sample is None:
            sample = builder.call_dict(uniform_name, {"target": result["m"]})
        return builder.finalize(
            {
                "mu": mu["mu"],
                "nu": transfer["nu"],
                "m": sample["target"],
                "flag_nu": result["succ"],
            }
        )
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
