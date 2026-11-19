# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Add.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Compose the shared ripple-add sequence as a typed, factorized program."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.component_predicate import control_predicate_program
from ..control.control_predicate import ControlRegister
from .addition_structure import addition_components


def addition_component_program(
    left_width: int,
    right_width: int | None = None,
    *,
    name: str = "main",
    left_dtype: RegisterType | None = None,
    right_dtype: RegisterType | None = None,
) -> ProgramRecord:
    """Construct ``a,b -> a,(a+b) mod 2**right_width`` in existing IR.

    Ports default to scalar uint words, ordered a then b, with MSB-first binding.
    Explicit integer descriptors retain their identities and must match widths.
    The source circuit zero-extends a narrower signed input before addition.
    Each AND owns its new carry until the matching measured erasure releases it.
    The existing predicate owner supplies physical AND bodies, clean allocation,
    feedback and erasure. The returned graph contains real executable functions;
    callers can retain AND leaves for tensor contraction or bind to append an op.
    No tensor, simulator, resource estimate or foreign source object is created.
    """
    if right_width is None:
        right_width = left_width
    if (
        type(left_width) is not int
        or type(right_width) is not int
        or not 1 <= left_width <= right_width
    ):
        raise ValueError("Add widths must be positive integers with left <= right")
    a_dtype = RegisterType("uint", left_width) if left_dtype is None else left_dtype
    b_dtype = RegisterType("uint", right_width) if right_dtype is None else right_dtype
    for dtype, width in ((a_dtype, left_width), (b_dtype, right_width)):
        if executable_register_width(dtype) != width or dtype.encoding not in {
            "uint",
            "signed",
            "montgomery",
        }:
            raise ValueError("Add requires integer descriptors matching its widths")
    if right_width == 1:
        signature = Signature((Register("a", a_dtype), Register("b", b_dtype)))
        cnot = bit_gate_function(
            "add_cx",
            (Register("ctrl", a_dtype), Register("target", b_dtype)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        )
        builder = ComponentProgramBuilder(signature, (cnot,), name=name)
        output = builder.call_dict(
            "add_cx", {"ctrl": builder.inputs["a"], "target": builder.inputs["b"]}
        )
        return builder.finalize({"a": output["ctrl"], "b": output["target"]})
    bit = RegisterType("bit", 1)
    ctrl = (Register("ctrl", bit, (2,)),)
    compute = control_predicate_program(
        ctrl, (ControlRegister(1, (1, 1)),), name="add_and"
    )
    erase = control_predicate_program(
        ctrl, (ControlRegister(1, (1, 1)),), name="add_and_adjoint", uncompute=True
    )
    cnot = bit_gate_function(
        "add_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    signature = Signature(
        (
            Register("a", a_dtype),
            Register("b", b_dtype),
        )
    )
    builder = ComponentProgramBuilder(
        signature, (cnot, *compute.functions, *erase.functions), name=name
    )
    a, b = builder.inputs["a"], builder.inputs["b"]
    assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
    wires: dict[int, RegisterBinding] = dict(
        enumerate((*builder.split(a), *builder.split(b)))
    )
    left = tuple(range(left_width))
    right = tuple(range(left_width, left_width + right_width))
    carries = tuple(range(left_width + right_width, left_width + 2 * right_width - 1))
    for kind, addresses in addition_components(left, right, carries):
        x, y = addresses[:2]
        if kind == "cx":
            result = builder.call_dict("add_cx", {"ctrl": wires[x], "target": wires[y]})
            cx, target = result["ctrl"], result["target"]
            assert isinstance(cx, RegisterBinding) and isinstance(
                target, RegisterBinding
            )
            wires[x], wires[y] = cx, target
            continue
        controls = np.array([wires[x], wires[y]], dtype=object)
        if kind == "and":
            result = builder.call_dict("add_and", {"ctrl": controls})
            target = result["target"]
            assert isinstance(target, RegisterBinding)
            wires[addresses[2]] = target
        else:
            result = builder.call_dict(
                "add_and_adjoint", {"ctrl": controls, "target": wires.pop(addresses[2])}
            )
        returned = np.asarray(result["ctrl"])
        wires[x], wires[y] = returned[0], returned[1]
    return builder.finalize(
        {
            "a": builder.join([wires[w] for w in left], a_dtype),
            "b": builder.join([wires[w] for w in right], b_dtype),
        }
    )
