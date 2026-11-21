# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LessThanEqual and GreaterThan.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Compose finite-width comparison trees with directional flag ownership."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..component_wire_graph import WireCall, wire_component_program
from ..control.conjunction_program import and_component_program, conjunction_program
from ._component_gates import comparison_bit_functions
from .less_equal import _tree, less_equal_requirements
from .tree_component import mixer_program, single_qubit_comparison_program


def less_equal_program(width: int, right_width: int | None = None) -> ProgramRecord:
    """XOR unsigned x<=y with source lazy words and exact reverse cleanup."""
    right_width = width if right_width is None else right_width
    less_equal_requirements(width, right_width)
    n, difference = min(width, right_width), abs(width - right_width)
    bit = quantum_bit_type()
    ports = {
        "x": tuple(range(width)),
        "y": tuple(range(width, width + right_width)),
        "target": (width + right_width,),
    }
    signature = Signature.build_from_dtypes(
        x=unsigned_register_type(width),
        y=unsigned_register_type(right_width),
        target=bit,
    )
    x, cx = (ProgramRecord((function,)) for function in comparison_bit_functions())
    calls: list[WireCall] = []
    cleanup: list[WireCall] = []
    cursor = width + right_width + 1
    prefix: int | None = None
    if difference:
        prefix = cursor
        cursor += 1
        controls = ports["x"][:-n] if width > right_width else ports["y"][:-n]
        if difference == 1:
            zero_calls: list[WireCall] = [
                (x, {"q": controls}),
                (cx, {"ctrl": controls, "target": (prefix,)}),
            ]
            zero_inverse = list(reversed(zero_calls))
        else:
            operands = {"ctrl": controls, "target": (prefix,)}
            if difference > 2:
                operands["junk"] = tuple(range(cursor, cursor + difference - 2))
                cursor += difference - 2
            if difference == 2:
                zero = and_component_program((0, 0), name="prefix_zero")
                inverse = and_component_program(
                    (0, 0), name="prefix_zero_erase", uncompute=True
                )
            else:
                zero = conjunction_program((0,) * difference, name="prefix_zero")
                inverse = conjunction_program(
                    (0,) * difference, name="prefix_zero_erase", uncompute=True
                )
            zero_calls, zero_inverse = [(zero, operands)], [(inverse, operands)]
        calls.extend(zero_calls)
        cleanup.extend(reversed(zero_inverse))
        if right_width > width:
            calls.extend(
                (
                    (x, {"q": ports["target"]}),
                    (cx, {"ctrl": (prefix,), "target": ports["target"]}),
                )
            )
    xs, ys = ports["x"][-n:], ports["y"][-n:]
    mixer, unmixer = mixer_program(), mixer_program(adjoint=True)
    for xi, yi in _tree(tuple(range(n)), tuple(range(n))):
        operands = {
            "x": tuple(xs[i] for i in xi),
            "y": tuple(ys[i] for i in yi),
            "ancilla": tuple(range(cursor, cursor + 3)),
        }
        cursor += 3
        calls.append((mixer, operands))
        cleanup.append((unmixer, operands))
    compare_ports = {
        "a": (xs[-1],),
        "b": (ys[-1],),
        "less_than": (cursor,),
        "greater_than": (cursor + 1,),
    }
    cursor += 2
    calls.append((single_qubit_comparison_program(), compare_ports))
    cleanup.append((single_qubit_comparison_program(adjoint=True), compare_ports))
    if prefix is None:
        calls.extend(
            (
                (x, {"q": ports["target"]}),
                (
                    cx,
                    {"ctrl": compare_ports["greater_than"], "target": ports["target"]},
                ),
            )
        )
    else:
        operands = {
            "ctrl": (prefix, compare_ports["greater_than"][0]),
            "target": (cursor,),
        }
        calls.append((and_component_program((1, 0), name="prefix_and"), operands))
        cleanup.append(
            (
                and_component_program((1, 0), name="prefix_and_erase", uncompute=True),
                operands,
            )
        )
        calls.append((cx, {"ctrl": (cursor,), "target": ports["target"]}))
    calls.extend(reversed(cleanup))
    return wire_component_program(signature, ports, ports, calls)


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
