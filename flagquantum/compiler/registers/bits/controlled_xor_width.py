# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 Controlled(Xor) and ControlledViaAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Bind controlled register-XOR widths to its existing atomic Toffoli emitter."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.resource_generalization import (
    BookkeepingCall,
    FixedTCall,
    ResourceCall,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..control.single_zero_control_width import single_zero_control_width
from ..resources.through_call_width import through_call_width
from .controlled_register_xor import controlled_xor_ir


def controlled_xor_width(
    call: ArithmeticCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Keep caller control and both big-endian data words live across each gate.

    Positive control retains source split/Toffoli/join ordering. Zero control
    shares the existing X/positive-child/X connections. The body is self-adjoint
    and owns zero work or classical slots. Symbolic widths use source fallback.
    """
    if call.operation != "controlled_xor":
        raise NotImplementedError("No controlled register-XOR width for this operation")
    width = call.widths[0]
    if call.control_value == 0:
        return single_zero_control_width(
            2 * width, replace(call, control_value=1, adjoint=False), child
        )
    if width == 0:
        raise ValueError("Controlled register-XOR width requires positive words")
    if type(width) is not int:
        return None
    source = tuple(range(1, width + 1))
    target = tuple(range(width + 1, 2 * width + 1))
    gates = controlled_xor_ir(0, source, target)

    def calls() -> Iterator[tuple[ResourceCall, tuple[int, ...]]]:
        yield BookkeepingCall("split", width), source
        yield BookkeepingCall("split", width), target
        for gate in gates:
            yield FixedTCall("toffoli"), gate.wires
        yield BookkeepingCall("join", width), source
        yield BookkeepingCall("join", width), target

    return through_call_width(2 * width + 1, calls(), child)
