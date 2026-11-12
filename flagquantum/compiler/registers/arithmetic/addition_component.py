# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Add.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Compose the shared ripple-add sequence as a typed, factorized program."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..control.component_predicate import _bit_function, control_predicate_program
from ..control.control_predicate import ControlRegister
from .addition_structure import addition_components


def addition_component_program(
    left_width: int, right_width: int | None = None, *, name: str = "main"
) -> ProgramRecord:
    """Construct unsigned ``a,b -> a,(a+b) mod 2**right_width`` in existing IR.

    Ports are scalar uint words, ordered a then b, with MSB-first bit binding.
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
    bit = RegisterType("bit", 1)
    ctrl = (Register("ctrl", bit, (2,)),)
    compute = control_predicate_program(
        ctrl, (ControlRegister(1, (1, 1)),), name="add_and"
    )
    erase = control_predicate_program(
        ctrl, (ControlRegister(1, (1, 1)),), name="add_and_adjoint", uncompute=True
    )
    cnot = _bit_function(
        "add_cx",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    signature = Signature(
        (
            Register("a", RegisterType("uint", left_width)),
            Register("b", RegisterType("uint", right_width)),
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
            "a": builder.join(
                [wires[w] for w in left], RegisterType("uint", left_width)
            ),
            "b": builder.join(
                [wires[w] for w in right], RegisterType("uint", right_width)
            ),
        }
    )
