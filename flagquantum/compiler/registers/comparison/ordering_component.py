# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanEqual and GreaterThan.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Compose finite-width comparison trees with directional flag ownership."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.conjunction_program import and_component_program, conjunction_program
from .less_equal import _tree, less_equal_requirements
from .tree_component import mixer_program, single_qubit_comparison_program


def less_equal_program(width: int, right_width: int | None = None) -> ProgramRecord:
    """XOR unsigned x<=y into a dirty target with all work cleaned.

    The existing source tree traversal owns its stage order. Mixer and final
    comparison flags remain live until the recorded reverse cleanup.
    """
    right_width = width if right_width is None else right_width
    less_equal_requirements(width, right_width)
    n = min(width, right_width)
    difference = abs(width - right_width)
    bit, word = quantum_bit_type(), unsigned_register_type(width)
    right_word = unsigned_register_type(right_width)
    x_gate = bit_gate_function(
        "compare_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    cx = bit_gate_function(
        "compare_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=word, y=right_word, target=bit), (x_gate, cx)
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    xs, ys = builder.split(x), builder.split(y)
    target = builder.inputs["target"]
    prefix: dict[str, BindingCollection] | None = None
    if difference:
        controls = xs[:-n] if width > right_width else ys[:-n]
        prefix = builder.call_program(
            prefix_zero_program(difference),
            {"ctrl": controls},
            entry="prefix_zero" if difference > 1 else "main",
        )
        if width > right_width:
            xs[:-n] = prefix["ctrl"]
        else:
            ys[:-n] = prefix["ctrl"]
            target = builder.call_dict("compare_x", {"q": target})["q"]
            copied = builder.call_dict(
                "compare_cx", {"ctrl": prefix["target"], "target": target}
            )
            prefix["target"], target = copied["ctrl"], copied["target"]
    x_suffix, y_suffix = xs[-n:], ys[-n:]
    stages: list[tuple[tuple[int, ...], tuple[int, ...], BindingCollection]] = []
    two_bits = unsigned_register_type(2)
    for xi, yi in _tree(tuple(range(n)), tuple(range(n))):
        result = builder.call_program(
            mixer_program(),
            {
                "x": builder.join(x_suffix[list(xi)], two_bits),
                "y": builder.join(y_suffix[list(yi)], two_bits),
            },
        )
        a, b = result["x"], result["y"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
        x_suffix[list(xi)], y_suffix[list(yi)] = builder.split(a), builder.split(b)
        stages.append((xi, yi, result["ancilla"]))
    flags = builder.call_program(
        single_qubit_comparison_program(), {"a": xs[-1], "b": ys[-1]}
    )
    xs[-1], ys[-1] = flags["a"], flags["b"]
    if prefix is None:
        target = builder.call_dict("compare_x", {"q": target})["q"]
        copied = builder.call_dict(
            "compare_cx", {"ctrl": flags["greater_than"], "target": target}
        )
        target, flags["greater_than"] = copied["target"], copied["ctrl"]
    else:
        combined = builder.call_program(
            and_component_program((1, 0), name="prefix_and"),
            {"ctrl": np.array([prefix["target"], flags["greater_than"]], dtype=object)},
            entry="prefix_and",
        )
        copied = builder.call_dict(
            "compare_cx", {"ctrl": combined["target"], "target": target}
        )
        target = copied["target"]
        combined["target"] = copied["ctrl"]
        erased = builder.call_program(
            and_component_program((1, 0), name="prefix_and_erase", uncompute=True),
            combined,
            entry="prefix_and_erase",
        )
        prefix["target"], flags["greater_than"] = np.asarray(erased["ctrl"])
    flags.update(a=xs[-1], b=ys[-1])
    restored = builder.call_program(
        single_qubit_comparison_program(adjoint=True), flags
    )
    xs[-1], ys[-1] = restored["a"], restored["b"]
    for xi, yi, ancilla in reversed(stages):
        result = builder.call_program(
            mixer_program(adjoint=True),
            {
                "x": builder.join(x_suffix[list(xi)], two_bits),
                "y": builder.join(y_suffix[list(yi)], two_bits),
                "ancilla": ancilla,
            },
        )
        a, b = result["x"], result["y"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
        x_suffix[list(xi)], y_suffix[list(yi)] = builder.split(a), builder.split(b)
    if prefix is not None:
        erased = builder.call_program(
            prefix_zero_program(difference, adjoint=True),
            prefix,
            entry="prefix_zero" if difference > 1 else "main",
        )
        if width > right_width:
            xs[:-n] = erased["ctrl"]
        else:
            ys[:-n] = erased["ctrl"]
    return builder.finalize(
        {
            "x": builder.join(xs, word),
            "y": builder.join(ys, right_word),
            "target": target,
        }
    )


def greater_than_program(
    width: int, right_width: int | None = None, *, adjoint: bool = False
) -> ProgramRecord:
    """XOR a>b with source LessThanEqual/X order and exact reversed adjoint."""
    if type(adjoint) is not bool:
        raise ValueError("Greater-than direction requires a boolean")
    right_width = width if right_width is None else right_width
    comparison = less_equal_program(width, right_width)
    bit, word = quantum_bit_type(), unsigned_register_type(width)
    right_word = unsigned_register_type(right_width)
    x_gate = bit_gate_function(
        "greater_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(a=word, b=right_word, target=bit), (x_gate,)
    )
    target = builder.inputs["target"]
    if adjoint:
        target = builder.call_dict("greater_x", {"q": target})["q"]
    result = builder.call_program(
        comparison,
        {"x": builder.inputs["a"], "y": builder.inputs["b"], "target": target},
    )
    target = result["target"]
    if not adjoint:
        target = builder.call_dict("greater_x", {"q": target})["q"]
    return builder.finalize({"a": result["x"], "b": result["y"], "target": target})


def prefix_zero_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Retain source prefix zero flags, including the one-bit modified control."""
    if type(width) is not int or width < 1 or type(adjoint) is not bool:
        raise ValueError(
            "Prefix comparison requires a positive width and boolean direction"
        )
    if width > 1:
        return conjunction_program((0,) * width, name="prefix_zero", uncompute=adjoint)
    bit = quantum_bit_type()
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    builder = ComponentProgramBuilder(
        Signature((Register("ctrl", bit, (1,)), Register("target", bit, side=side))),
        (
            bit_gate_function(
                "prefix_x",
                (Register("q", bit),),
                CircuitIR(1, (Instruction("x", (0,)),)),
            ),
            bit_gate_function(
                "prefix_cx",
                (Register("ctrl", bit), Register("target", bit)),
                CircuitIR(2, (Instruction("cx", (0, 1)),)),
            ),
        ),
    )
    control = np.asarray(builder.inputs["ctrl"])[0]
    target = builder.inputs["target"] if adjoint else builder.allocate(dtype=bit)
    if not adjoint:
        control = builder.call_dict("prefix_x", {"q": control})["q"]
    copied = builder.call_dict("prefix_cx", {"ctrl": control, "target": target})
    control, target = copied["ctrl"], copied["target"]
    if adjoint:
        control = builder.call_dict("prefix_x", {"q": control})["q"]
        assert isinstance(target, RegisterBinding)
        builder.free(target)
    outputs: dict[str, BindingCollection] = {"ctrl": np.array([control], dtype=object)}
    if not adjoint:
        outputs["target"] = target
    return builder.finalize(outputs)
