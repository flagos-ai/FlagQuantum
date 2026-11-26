# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LinearDepthGreaterThan.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Retain the source carry chain and its measured reverse computation."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..control.conjunction_program import and_component_program
from ..control.multi_control_component import multi_control_x_program
from ._component_gates import comparison_bit_functions
from .linear_comparison import linear_greater_requirements


def linear_greater_program(
    width: int, *, signed: bool = False, adjoint: bool = False
) -> ProgramRecord:
    """XOR the source finite-width difference sign into a dirty target.

    Inputs keep the source unsigned port types even in signed mode. Signed mode
    retains the original overflow behavior and the unsigned one-bit base case.
    All carry and unsigned sign work is owned and erased before return.
    The adjoint reverses exact child calls before measurement lowering.
    """
    if type(adjoint) is not bool:
        raise ValueError("Linear comparison direction must be Boolean")
    linear_greater_requirements(width, signed=signed)
    word, bit = unsigned_register_type(width), quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(a=word, b=word, target=bit),
        comparison_bit_functions(),
    )
    a, b = builder.inputs["a"], builder.inputs["b"]
    assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    target = builder.inputs["target"]
    if width == 1:
        result = builder.call_program(
            multi_control_x_program((1, 0)),
            {
                "controls": np.array([a, b], dtype=object),
                "target": target,
            },
        )
        base_controls = np.asarray(result["controls"])
        record = builder.finalize(
            {
                "a": base_controls[0],
                "b": base_controls[1],
                "target": result["target"],
            }
        )
        copy = multi_control_x_program((1, 0))
        return adjoint_component_calls(record, ((copy, copy),)) if adjoint else record
    true_width = width if signed else width + 1
    wide = unsigned_register_type(true_width)
    if not signed:
        prefix = builder.allocate()
        a = builder.join([prefix, *builder.split(a)], wide)
        prefix = builder.allocate()
        b = builder.join([prefix, *builder.split(b)], wide)
    bs = builder.split(b)
    for i in range(true_width):
        bs[i] = builder.call_dict("comparison_x", {"q": bs[i]})["q"]
    xs = builder.split(a)
    carries: list[BindingCollection] = []
    controls: list[BindingCollection] = []
    compute = and_component_program((1, 1), name="carry_and")
    erase = and_component_program((1, 1), name="carry_erase", uncompute=True)
    for i in range(true_width - 1):
        index = -1 - i
        if i:
            result = builder.call_dict(
                "comparison_cx", {"ctrl": carries[-1], "target": bs[index]}
            )
            carries[-1], bs[index] = result["ctrl"], result["target"]
            result = builder.call_dict(
                "comparison_cx", {"ctrl": carries[-1], "target": xs[index]}
            )
            carries[-1], xs[index] = result["ctrl"], result["target"]
        result = builder.call_program(
            compute,
            {"ctrl": np.array([bs[index], xs[index]], dtype=object)},
            entry="carry_and",
        )
        controls.append(result["ctrl"])
        carries.append(result["target"])
        if i:
            result = builder.call_dict(
                "comparison_cx", {"ctrl": carries[-2], "target": carries[-1]}
            )
            carries[-2], carries[-1] = result["ctrl"], result["target"]
    result = builder.call_dict("comparison_cx", {"ctrl": carries[-1], "target": xs[0]})
    carries[-1], xs[0] = result["ctrl"], result["target"]
    result = builder.call_dict("comparison_cx", {"ctrl": bs[0], "target": xs[0]})
    bs[0], xs[0] = result["ctrl"], result["target"]
    xs[0] = builder.call_dict("comparison_x", {"q": xs[0]})["q"]
    result = builder.call_dict("comparison_cx", {"ctrl": xs[0], "target": target})
    xs[0], target = result["ctrl"], result["target"]
    xs[0] = builder.call_dict("comparison_x", {"q": xs[0]})["q"]
    result = builder.call_dict("comparison_cx", {"ctrl": bs[0], "target": xs[0]})
    bs[0], xs[0] = result["ctrl"], result["target"]
    result = builder.call_dict("comparison_cx", {"ctrl": carries[-1], "target": xs[0]})
    carries[-1], xs[0] = result["ctrl"], result["target"]
    for i in range(true_width - 1):
        carry = carries.pop()
        if carries:
            result = builder.call_dict(
                "comparison_cx", {"ctrl": carries[-1], "target": carry}
            )
            carries[-1], carry = result["ctrl"], result["target"]
        result = builder.call_program(
            erase, {"ctrl": controls.pop(), "target": carry}, entry="carry_erase"
        )
        bs[i + 1], xs[i + 1] = np.asarray(result["ctrl"])
        if carries:
            result = builder.call_dict(
                "comparison_cx", {"ctrl": carries[-1], "target": xs[i + 1]}
            )
            carries[-1], xs[i + 1] = result["ctrl"], result["target"]
            result = builder.call_dict(
                "comparison_cx", {"ctrl": carries[-1], "target": bs[i + 1]}
            )
            carries[-1], bs[i + 1] = result["ctrl"], result["target"]
    for i in range(true_width):
        bs[i] = builder.call_dict("comparison_x", {"q": bs[i]})["q"]
    a, b = builder.join(xs, wide), builder.join(bs, wide)
    if not signed:
        xs = builder.split(a)
        a = builder.join(xs[1:], word)
        builder.free(xs[0])
        bs = builder.split(b)
        b = builder.join(bs[1:], word)
        builder.free(bs[0])
    record = builder.finalize({"a": a, "b": b, "target": target})
    if not adjoint:
        return record
    x_gate, cx_gate = (
        ProgramRecord((function,)) for function in comparison_bit_functions()
    )
    return adjoint_component_calls(
        record,
        ((compute, erase), (erase, compute), (x_gate, x_gate), (cx_gate, cx_gate)),
    )
