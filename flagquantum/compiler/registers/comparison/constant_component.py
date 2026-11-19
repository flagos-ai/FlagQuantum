# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanConstant.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Retain the source prefix-equality scan and its measured reverse cleanup."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..control.conjunction_program import and_component_program
from ._component_gates import comparison_bit_functions
from .less_than import less_than_constant_workspace_size


def less_than_constant_program(width: int, threshold: int) -> ProgramRecord:
    """XOR unsigned x<threshold into a dirty target with source And history.

    The equality seed and every prefix flag start clean. All are restored by
    reversing the recorded scan. The source out-of-range upper threshold is
    one X, including no workspace. Negative thresholds remain rejected.
    """
    less_than_constant_workspace_size(width, threshold)
    dtype, bit = unsigned_register_type(width), quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=dtype, target=bit), comparison_bit_functions()
    )
    word, target = builder.inputs["x"], builder.inputs["target"]
    assert isinstance(word, RegisterBinding)
    if threshold >= 1 << width:
        target = builder.call_dict("comparison_x", {"q": target})["q"]
        return builder.finalize({"x": word, "target": target})
    xs = builder.split(word)
    equal: BindingCollection = builder.allocate()
    equal = builder.call_dict("comparison_x", {"q": equal})["q"]
    flags = []
    pattern = tuple((threshold >> shift) & 1 for shift in reversed(range(width)))
    for index, value in enumerate(pattern):
        if value:
            xs[index] = builder.call_dict("comparison_x", {"q": xs[index]})["q"]
        computed = builder.call_program(
            and_component_program((1, 1), name="prefix_and"),
            {"ctrl": np.array([xs[index], equal], dtype=object)},
            entry="prefix_and",
        )
        xs[index], equal = np.asarray(computed["ctrl"])
        flag = computed["target"]
        if value:
            copied = builder.call_dict(
                "comparison_cx", {"ctrl": flag, "target": target}
            )
            flag, target = copied["ctrl"], copied["target"]
        copied = builder.call_dict("comparison_cx", {"ctrl": flag, "target": equal})
        flags.append(copied["ctrl"])
        equal = copied["target"]
    for index in reversed(range(width)):
        copied = builder.call_dict(
            "comparison_cx", {"ctrl": flags[index], "target": equal}
        )
        equal = copied["target"]
        restored = builder.call_program(
            and_component_program((1, 1), name="prefix_erase", uncompute=True),
            {
                "ctrl": np.array([xs[index], equal], dtype=object),
                "target": copied["ctrl"],
            },
            entry="prefix_erase",
        )
        xs[index], equal = np.asarray(restored["ctrl"])
        if pattern[index]:
            xs[index] = builder.call_dict("comparison_x", {"q": xs[index]})["q"]
    equal = builder.call_dict("comparison_x", {"q": equal})["q"]
    assert isinstance(equal, RegisterBinding)
    builder.free(equal)
    return builder.finalize({"x": builder.join(xs, dtype), "target": target})
