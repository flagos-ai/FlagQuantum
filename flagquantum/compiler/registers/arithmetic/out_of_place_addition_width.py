# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 OutOfPlaceAdder and Cirq import lifetimes.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Bind sum-output creation and consumption to native logical width queries."""

from __future__ import annotations

from collections.abc import Callable

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
from .out_of_place_addition_structure import out_of_place_addition_components


def out_of_place_addition_width(
    call: ArithmeticCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Own only the low output allocation and each AND-created output bit.

    Input words stay caller-owned. Adjoint consumes the supplied sum, then
    frees its low bit. Views are introduced on first use, as in source import.
    Non-Python dimensions retain signature/callee fallback and zero-count visits.
    """
    if call.operation != "out_of_place_add":
        raise NotImplementedError("No out-of-place addition width for this operation")
    width = call.widths[0]
    if type(width) is not int:
        return None
    if width < 1:
        raise ValueError("Out-of-place addition width requires positive words")
    left = tuple(range(width))
    right = tuple(range(width, 2 * width))
    output = tuple(range(2 * width, 3 * width + int(call.include_most_significant_bit)))
    words = (left, right, output) if call.adjoint else (left, right)
    producers = dict.fromkeys((wire for word in words for wire in word), 0)
    pending_views = {wire: word for word in words for wire in word if len(word) > 1}
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
        kind = (
            key.opcode
            if isinstance(key, BookkeepingCall) and key.opcode in {"allocate", "free"}
            else "component"
        )
        steps.append(WidthStep(kind))
        keys.append(key)

    for kind, wires in out_of_place_addition_components(
        left,
        right,
        output,
        include_msb=call.include_most_significant_bit,
        adjoint=call.adjoint,
    ):
        inputs = wires[:2] if kind == "and" else wires
        for wire in inputs:
            if wire in pending_views:
                word = pending_views[wire]
                add(BookkeepingCall("split", len(word)), word, word)
                for bit in word:
                    del pending_views[bit]
            if wire not in producers:
                add(BookkeepingCall("allocate", 1), (), (wire,))
        if kind == "cx":
            add(CliffordCall("cx", 2), wires, wires)
        else:
            add(
                ConjunctionCall((1, 1), call.adjoint),
                inputs,
                wires[:2] if call.adjoint else wires,
            )
    for word in (left, right):
        if len(word) > 1:
            add(BookkeepingCall("join", len(word)), word, word)
    if call.adjoint:
        add(BookkeepingCall("free", 1), (output[-1],), ())
    elif len(output) > 1:
        add(BookkeepingCall("join", len(output)), output, output)
    end = len(steps)
    connections.extend(WidthConnection(source, end, 1) for source in producers.values())
    steps.append(WidthStep("output"))
    for group in schedule_width_steps(steps, connections):
        for index in group:
            if 0 < index < end:
                steps[index] = WidthStep(steps[index].kind, child(keys[index - 1]))
    return estimate_connection_width(steps, connections)
