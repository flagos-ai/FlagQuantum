# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HasDuplicates.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/has_duplicates.py

"""Retain adjacent half-comparison histories and the source inverse aliases."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import RegisterType
from ..comparison._component_gates import comparison_bit_functions
from ..comparison.half_component import half_comparison_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..control.multi_control_component import multi_control_x_program
from .sorted_duplicates import (
    sorted_duplicate_requirements,
    sorted_duplicates_signature,
)


def sorted_duplicates_program(
    length: int, dtype: RegisterType, *, controlled: bool = False
) -> ProgramRecord:
    """Store the original adjacent comparison/MCX/reverse-history graph.

    xs is an MSB-first shaped array and flag may be dirty. The optional ctrl is
    active on one. Original inverse half-LT calls return swapped input aliases.
    These aliases are propagated even when a later erasure becomes invalid.
    The source uncontrolled singleton has no executable decomposition.
    """
    width = executable_register_width(dtype)
    sorted_duplicate_requirements(length, width, controlled=controlled)
    if dtype.encoding not in {"uint", "signed"}:
        raise ValueError(
            "Duplicate comparison requires unsigned or signed integer words"
        )
    if length == 1 and not controlled:
        raise ValueError(
            "Source duplicate detection cannot decompose an uncontrolled singleton"
        )
    if length > 1:
        forward = half_comparison_program(dtype, "lt")
        inverse = half_comparison_program(dtype, "lt", adjoint=True)
    builder = ComponentProgramBuilder(
        sorted_duplicates_signature(length, dtype, controlled=controlled),
        comparison_bit_functions(),
    )
    xs = np.asarray(builder.inputs["xs"])
    flag = builder.inputs["flag"]
    histories: list[BindingCollection] = []
    flags: list[BindingCollection] = [builder.inputs["ctrl"]] if controlled else []
    for i in range(1, length):
        result = builder.call_program(forward, {"a": xs[i - 1], "b": xs[i]})
        xs[i - 1], xs[i] = result["a"], result["b"]
        histories.append(result["c"])
        flags.append(result["target"])
    result = builder.call_program(
        multi_control_x_program((1,) * len(flags)),
        {"controls": np.array(flags, dtype=object), "target": flag},
    )
    flags = list(np.asarray(result["controls"]))
    flag = result["target"]
    if controlled:
        copied = builder.call_dict("comparison_cx", {"ctrl": flags[0], "target": flag})
        flags[0], flag = copied["ctrl"], copied["target"]
    else:
        flag = builder.call_dict("comparison_x", {"q": flag})["q"]
    for i in reversed(range(1, length)):
        result = builder.call_program(
            inverse,
            {"a": xs[i - 1], "b": xs[i], "c": histories.pop(), "target": flags.pop()},
        )
        xs[i - 1], xs[i] = result["a"], result["b"]
    outputs: dict[str, BindingCollection] = {"xs": xs, "flag": flag}
    if controlled:
        outputs["ctrl"] = flags.pop()
    assert not flags and not histories
    return builder.finalize(outputs)
