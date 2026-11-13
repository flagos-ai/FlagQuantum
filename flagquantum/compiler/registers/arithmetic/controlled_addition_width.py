# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 CAdd and its generic adjoint.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/controlled_addition.py

"""Bind CAdd's controlled load, Add child and erasure to width connections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.resource_generalization import (
    BookkeepingCall,
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


def controlled_addition_width(
    call: ArithmeticCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Keep caller control and big-endian words live through loaded-word erasure.

    AND creates each loaded bit and AND adjoint consumes it. The Add child owns
    its carries. View keys retain the existing width-only normalization, while
    cast keys retain width and input/output encodings. Symbolic dimensions use
    the signature/callee fallback. Reversing edges gives the exact adjoint
    lifetime graph, including the reversed cast and word-view order.
    """
    if call.operation != "controlled_add":
        raise NotImplementedError("No CAdd width decomposition for this operation")
    a_width, b_width = call.widths
    if type(a_width) is not int or type(b_width) is not int:
        return None
    if not 1 <= a_width <= b_width:
        raise ValueError("CAdd width requires positive words with left <= right")
    control = (0,)
    left = tuple(range(1, a_width + 1))
    right = tuple(range(a_width + 1, a_width + b_width + 1))
    loaded = tuple(range(a_width + b_width + 1, 2 * a_width + b_width + 1))
    producers = dict.fromkeys((*control, *left, *right), 0)
    steps = [WidthStep("input")]
    connections: list[WidthConnection] = []
    keys: list[ArithmeticCall | BookkeepingCall | ConjunctionCall] = []

    def add(
        key: ArithmeticCall | BookkeepingCall | ConjunctionCall,
        inputs: tuple[int, ...],
        outputs: tuple[int, ...],
    ) -> None:
        index = len(steps)
        for wire in inputs:
            connections.append(WidthConnection(producers.pop(wire), index, 1))
        producers.update(dict.fromkeys(outputs, index))
        steps.append(WidthStep("component"))
        keys.append(key)

    add(BookkeepingCall("split", a_width), left, left)
    add(BookkeepingCall("split", 1), control, control)
    for source, target in zip(left, loaded, strict=True):
        add(ConjunctionCall((call.control_value, 1)), (0, source), (0, source, target))
    add(BookkeepingCall("join", a_width), loaded, loaded)
    add(
        BookkeepingCall("cast", (a_width, "unsigned", call.encodings[0])),
        loaded,
        loaded,
    )
    add(
        ArithmeticCall("add", call.widths, call.encodings),
        (*loaded, *right),
        (*loaded, *right),
    )
    add(BookkeepingCall("split", a_width), loaded, loaded)
    for source, target in reversed(tuple(zip(left, loaded, strict=True))):
        add(
            ConjunctionCall((call.control_value, 1), True),
            (0, source, target),
            (0, source),
        )
    add(BookkeepingCall("join", a_width), left, left)
    add(BookkeepingCall("join", 1), control, control)
    output = len(steps)
    connections.extend(
        WidthConnection(source, output, 1) for source in producers.values()
    )
    steps.append(WidthStep("output"))
    if call.adjoint:
        connections = [
            WidthConnection(output - edge.target, output - edge.source, edge.qubits)
            for edge in connections
        ]
        keys = [_adjoint_key(key) for key in reversed(keys)]
    for group in schedule_width_steps(steps, connections):
        for index in group:
            if 0 < index < output:
                steps[index] = WidthStep("component", child(keys[index - 1]))
    return estimate_connection_width(steps, connections)


def _adjoint_key(
    key: ArithmeticCall | BookkeepingCall | ConjunctionCall,
) -> ArithmeticCall | BookkeepingCall | ConjunctionCall:
    if isinstance(key, BookkeepingCall):
        if key.opcode == "cast":
            assert isinstance(key.layout, tuple)
            width, inp, out = key.layout
            return BookkeepingCall("cast", (width, out, inp))
        return BookkeepingCall("join" if key.opcode == "split" else "split", key.layout)
    return replace(key, adjoint=not key.adjoint)
