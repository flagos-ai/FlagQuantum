# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanEqual and GreaterThan.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Compose equal-width comparison trees with directional flag ownership."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .less_equal import _tree, less_equal_requirements
from .tree_component import mixer_program, single_qubit_comparison_program


def less_equal_program(width: int) -> ProgramRecord:
    """XOR equal-width unsigned x<=y into a dirty target with all work cleaned.

    The existing source tree traversal owns its stage order. Mixer and final
    comparison flags remain live until the recorded reverse cleanup.
    """
    less_equal_requirements(width, width)
    bit, word = quantum_bit_type(), unsigned_register_type(width)
    x_gate = bit_gate_function(
        "compare_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    cx = bit_gate_function(
        "compare_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=word, y=word, target=bit), (x_gate, cx)
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
    xs, ys = builder.split(x), builder.split(y)
    stages: list[tuple[tuple[int, ...], tuple[int, ...], BindingCollection]] = []
    two_bits = unsigned_register_type(2)
    for xi, yi in _tree(tuple(range(width)), tuple(range(width))):
        result = builder.call_program(
            mixer_program(),
            {
                "x": builder.join(xs[list(xi)], two_bits),
                "y": builder.join(ys[list(yi)], two_bits),
            },
        )
        a, b = result["x"], result["y"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
        xs[list(xi)], ys[list(yi)] = builder.split(a), builder.split(b)
        stages.append((xi, yi, result["ancilla"]))
    flags = builder.call_program(
        single_qubit_comparison_program(), {"a": xs[-1], "b": ys[-1]}
    )
    xs[-1], ys[-1] = flags["a"], flags["b"]
    target = builder.call_dict("compare_x", {"q": builder.inputs["target"]})["q"]
    copied = builder.call_dict(
        "compare_cx", {"ctrl": flags["greater_than"], "target": target}
    )
    flags.update(a=xs[-1], b=ys[-1], greater_than=copied["ctrl"])
    restored = builder.call_program(
        single_qubit_comparison_program(adjoint=True), flags
    )
    xs[-1], ys[-1] = restored["a"], restored["b"]
    for xi, yi, ancilla in reversed(stages):
        result = builder.call_program(
            mixer_program(adjoint=True),
            {
                "x": builder.join(xs[list(xi)], two_bits),
                "y": builder.join(ys[list(yi)], two_bits),
                "ancilla": ancilla,
            },
        )
        a, b = result["x"], result["y"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
        xs[list(xi)], ys[list(yi)] = builder.split(a), builder.split(b)
    return builder.finalize(
        {
            "x": builder.join(xs, word),
            "y": builder.join(ys, word),
            "target": copied["target"],
        }
    )


def greater_than_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """XOR a>b with source LessThanEqual/X order and exact reversed adjoint."""
    if type(adjoint) is not bool:
        raise ValueError("Greater-than direction requires a boolean")
    comparison = less_equal_program(width)
    bit, word = quantum_bit_type(), unsigned_register_type(width)
    x_gate = bit_gate_function(
        "greater_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(a=word, b=word, target=bit), (x_gate,)
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
