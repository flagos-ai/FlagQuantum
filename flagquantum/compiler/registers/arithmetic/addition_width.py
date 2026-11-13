# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 Add and its generic adjoint.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Bind the shared Add component sequence to carry-owning width connections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ResourceCall,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
    schedule_width_steps,
)
from .addition_structure import addition_components


def addition_width(
    call: ArithmeticCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Keep caller words live while each AND owns its carry until erasure.

    Scalar one-bit views remain atomic, matching the original Cirq import.
    Adjoint traverses the same logical wire identities in reverse dependency
    order. Symbolic dimensions retain the source signature/callee fallback.
    """
    if call.operation != "add":
        raise NotImplementedError("No Add width decomposition for this operation")
    if call.always:
        return child(replace(call, always=False))
    a_width, b_width = call.widths
    if type(a_width) is not int or type(b_width) is not int:
        return None
    if not 1 <= a_width <= b_width:
        raise ValueError("Add width requires positive words with left <= right")
    left = tuple(range(a_width))
    right = tuple(range(a_width, a_width + b_width))
    carries = tuple(range(a_width + b_width, a_width + 2 * b_width - 1))
    producers = dict.fromkeys((*left, *right), 0)
    steps = [WidthStep("input")]
    connections: list[WidthConnection] = []
    keys: list[ResourceCall] = []

    def add(
        key: ResourceCall, inputs: tuple[int, ...], outputs: tuple[int, ...]
    ) -> None:
        index = len(steps)
        for wire in inputs:
            connections.append(WidthConnection(producers.pop(wire), index, 1))
        producers.update(dict.fromkeys(outputs, index))
        steps.append(WidthStep("component"))
        keys.append(key)

    words = (right, left) if call.adjoint else (left, right)
    for word in words:
        if len(word) > 1:
            add(BookkeepingCall("split", len(word)), word, word)
    for kind, wires in addition_components(left, right, carries, adjoint=call.adjoint):
        if kind == "cx":
            add(CliffordCall("cx", 2), wires, wires)
        elif kind == "and":
            add(ConjunctionCall((1, 1)), wires[:2], wires)
        else:
            add(ConjunctionCall((1, 1), True), wires, wires[:2])
    for word in words:
        if len(word) > 1:
            add(BookkeepingCall("join", len(word)), word, word)
    output = len(steps)
    connections.extend(
        WidthConnection(source, output, 1) for source in producers.values()
    )
    steps.append(WidthStep("output"))
    for group in schedule_width_steps(steps, connections):
        for index in group:
            if 0 < index < output:
                steps[index] = WidthStep("component", child(keys[index - 1]))
    return estimate_connection_width(steps, connections)
