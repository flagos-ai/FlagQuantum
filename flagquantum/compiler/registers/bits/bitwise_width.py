# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Xor, XorK and BitwiseNot.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py

"""Project shared bitwise emitters and OnEach into native width connections."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    OnEachCall,
    ResourceCall,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..arithmetic.arithmetic_declarations import arithmetic_callees
from ..resources.through_call_width import through_call_width
from .bitwise import xor_constant_ir, xor_ir

BITWISE_WIDTH_OPERATIONS = ("xor", "xor_constant", "bitwise_not")


def bitwise_width(
    call: ArithmeticCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Keep big-endian through words and source self-adjoint child identities.

    These operations have no work registers or classical slots. Constants use
    the existing signed range validation before binding their unsigned pattern.
    Non-Python widths or constants retain the original symbolic fallback.
    """
    if call.operation not in BITWISE_WIDTH_OPERATIONS:
        raise NotImplementedError("No bitwise width decomposition for this operation")
    width = call.widths[0]
    if width == 0:
        raise ValueError("Bitwise width requires nonempty registers")
    if call.operation == "bitwise_not":
        return child(OnEachCall(CliffordCall("x", 1), width))
    if type(width) is not int:
        return None
    if call.operation == "xor_constant" and type(call.constant) is not int:
        return None
    word = tuple(range(width))
    if call.operation == "xor":
        target = tuple(range(width, 2 * width))
        gates = xor_ir(word, target)
    else:
        # Reuse the source dtype range checks, including signed constants.
        arithmetic_callees(call)
        assert isinstance(call.constant, int)
        target = ()
        gates = xor_constant_ir(word, call.constant % (1 << width))

    def calls() -> Iterator[tuple[ResourceCall, tuple[int, ...]]]:
        yield BookkeepingCall("split", width), word
        if target:
            yield BookkeepingCall("split", width), target
        for gate in gates:
            yield CliffordCall("cx" if target else "x", len(gate.wires)), gate.wires
        yield BookkeepingCall("join", width), word
        if target:
            yield BookkeepingCall("join", width), target

    return through_call_width(len(word) + len(target), calls(), child)
