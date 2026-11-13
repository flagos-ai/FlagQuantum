# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 Controlled(XorK) and ControlledViaAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Bind controlled XorK width to shared CNOT emission and caller-word views."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ResourceCall,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..arithmetic.arithmetic_declarations import arithmetic_callees
from ..control.single_zero_control_width import single_zero_control_width
from ..resources.through_call_width import through_call_width
from .bitwise import controlled_xor_constant_ir


def controlled_constant_width(
    call: ArithmeticCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Retain big-endian target and caller control across literal source CNOTs."""
    if call.operation != "controlled_xor_constant":
        raise NotImplementedError(
            "No controlled XorK width decomposition for this operation"
        )
    width = call.widths[0]
    if call.control_value == 0:
        return single_zero_control_width(
            width, replace(call, control_value=1, adjoint=False), child
        )
    if width == 0:
        raise ValueError("Controlled XorK width requires a positive word")
    if type(width) is not int or type(call.constant) is not int:
        return None
    arithmetic_callees(call)
    assert isinstance(call.constant, int)
    word = tuple(range(1, width + 1))
    gates = controlled_xor_constant_ir(0, word, call.constant % (1 << width))

    def calls() -> Iterator[tuple[ResourceCall, tuple[int, ...]]]:
        yield BookkeepingCall("split", width), word
        for gate in gates:
            yield CliffordCall("cx", 2), gate.wires
        yield BookkeepingCall("join", width), word

    return through_call_width(width + 1, calls(), child)
