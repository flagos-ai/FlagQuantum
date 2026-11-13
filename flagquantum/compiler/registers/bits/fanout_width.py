# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiTargetCNOT.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_target_cnot.py

"""Derive fanout width from the native layered CNOT stream."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from ..._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    FanoutCall,
    ResourceCall,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..resources.through_call_width import through_call_width
from .bitwise import fanout_ir


def fanout_width(
    call: FanoutCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Retain atomic one-bit views and explicit fanout adjoint identity."""
    width = call.width
    if type(width) is not int:
        return None
    if width < 1:
        raise ValueError("Fanout width requires nonempty targets")
    targets = tuple(range(1, width + 1))
    gates = fanout_ir(0, targets)
    if call.adjoint:
        gates = tuple(reversed(gates))

    def calls() -> Iterator[tuple[ResourceCall, tuple[int, ...]]]:
        if width > 1:
            yield BookkeepingCall("split", width), targets
        for gate in gates:
            yield CliffordCall("cx", 2), gate.wires
        if width > 1:
            yield BookkeepingCall("join", width), targets

    return through_call_width(width + 1, calls(), child)
