# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BiQubitsMixer and SingleQubitCompare.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py

"""Own directional comparison flags over shared typed CX and And components."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_wire_graph import WireCall, wire_component_program
from ..control.conjunction_program import and_component_program
from ._component_gates import comparison_bit_functions


def mixer_program(*, adjoint: bool = False) -> ProgramRecord:
    """Mix two two-bit words, producing or consuming three owned flag qubits.

    The reverse source call sequence erases both And flags by measurement and
    consumes the remaining clean seed. Input and returned words are MSB-first.
    """
    if type(adjoint) is not bool:
        raise ValueError("Comparison adjoint flag must be boolean")
    x, cx = (ProgramRecord((function,)) for function in comparison_bit_functions())
    conjunction = and_component_program((1, 1), name="mixer_and", uncompute=adjoint)
    calls: list[WireCall] = [
        (x, {"q": (4,)}),
        (cx, {"ctrl": (2,), "target": (0,)}),
        (cx, {"ctrl": (3,), "target": (1,)}),
    ]
    for control, a, b, auxiliary in ((0, 1, 4, 5), (0, 2, 3, 6)):
        calls.extend(
            (
                (cx, {"ctrl": (a,), "target": (b,)}),
                (conjunction, {"ctrl": (control, b), "target": (auxiliary,)}),
                (cx, {"ctrl": (auxiliary,), "target": (a,)}),
                (cx, {"ctrl": (a,), "target": (b,)}),
            )
        )
    calls.append((cx, {"ctrl": (3,), "target": (1,)}))
    word, work = unsigned_register_type(2), opaque_register_type(3)
    signature = Signature(
        (
            Register("x", word),
            Register("y", word),
            Register(
                "ancilla",
                work,
                side=RegisterSide.LEFT if adjoint else RegisterSide.RIGHT,
            ),
        )
    )
    ports = {"x": (0, 1), "y": (2, 3)}
    flags = {"ancilla": (4, 5, 6)}
    return wire_component_program(
        signature,
        {**ports, **(flags if adjoint else {})},
        {**ports, **({} if adjoint else flags)},
        reversed(calls) if adjoint else calls,
    )


def single_qubit_comparison_program(*, adjoint: bool = False) -> ProgramRecord:
    """Produce comparison flags and equality-in-b, or consume their exact image."""
    if type(adjoint) is not bool:
        raise ValueError("Comparison adjoint flag must be boolean")
    x, cx = (ProgramRecord((function,)) for function in comparison_bit_functions())
    conjunction = and_component_program(
        (0, 1), name="single_compare_and", uncompute=adjoint
    )
    calls: list[WireCall] = [
        (conjunction, {"ctrl": (0, 1), "target": (2,)}),
        (cx, {"ctrl": (2,), "target": (3,)}),
        (cx, {"ctrl": (1,), "target": (3,)}),
        (cx, {"ctrl": (0,), "target": (1,)}),
        (cx, {"ctrl": (0,), "target": (3,)}),
        (x, {"q": (1,)}),
    ]
    bit = quantum_bit_type()
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    signature = Signature(
        (
            Register("a", bit),
            Register("b", bit),
            Register("less_than", bit, side=side),
            Register("greater_than", bit, side=side),
        )
    )
    ports, flags = {"a": (0,), "b": (1,)}, {"less_than": (2,), "greater_than": (3,)}
    return wire_component_program(
        signature,
        {**ports, **(flags if adjoint else {})},
        {**ports, **({} if adjoint else flags)},
        reversed(calls) if adjoint else calls,
    )
