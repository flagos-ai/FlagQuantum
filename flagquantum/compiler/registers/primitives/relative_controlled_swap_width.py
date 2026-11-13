# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CSwapApprox.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/cswap_approx.py

"""Project native relative-swap segments and the retained fanout boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Literal

from ....core.ir import Instruction
from ..._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ControlledSwapCall,
    FanoutCall,
    PhaseCall,
    ResourceCall,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..resources.through_call_width import through_call_width
from .relative_controlled_swap import _INVERSE_PHASE, _relative_swap_parts


def relative_controlled_swap_width(
    call: ControlledSwapCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Use the native literal segments in source component connection order."""
    width = call.width
    if type(width) is not int:
        return None
    if width < 1:
        raise ValueError("Relative controlled swap width requires nonempty words")
    left = tuple(range(1, width + 1))
    right = tuple(range(width + 1, 2 * width + 1))
    before, after = _relative_swap_parts(left, right)
    if call.adjoint:
        before, after = _adjoint(after), _adjoint(before)

    def calls() -> Iterator[tuple[ResourceCall, tuple[int, ...]]]:
        if width > 1:
            # The first CNOT consumes the right word before the left word.
            yield BookkeepingCall("split", width), right
            yield BookkeepingCall("split", width), left
        for gate in before:
            yield _gate_key(gate), gate.wires
        if width > 1:
            yield BookkeepingCall("join", width), right
        yield FanoutCall(width, call.adjoint), (0, *right)
        if width > 1:
            yield BookkeepingCall("split", width), right
        for gate in after:
            yield _gate_key(gate), gate.wires
        if width > 1:
            yield BookkeepingCall("join", width), left
            yield BookkeepingCall("join", width), right

    return through_call_width(2 * width + 1, calls(), child)


def _adjoint(gates: tuple[Instruction, ...]) -> tuple[Instruction, ...]:
    return tuple(
        Instruction(_INVERSE_PHASE.get(gate.name, gate.name), gate.wires)
        for gate in reversed(gates)
    )


def _gate_key(gate: Instruction) -> ResourceCall:
    if gate.name in {"s", "sdg", "t", "tdg"}:
        opcode: Literal["s", "t"] = "s" if gate.name.startswith("s") else "t"
        return PhaseCall(opcode, gate.name.endswith("dg"))
    if gate.name == "h":
        return CliffordCall("h", 1)
    if gate.name == "cx":
        return CliffordCall("cx", 2)
    raise ValueError(f"Relative swap width cannot bind gate {gate.name!r}")
