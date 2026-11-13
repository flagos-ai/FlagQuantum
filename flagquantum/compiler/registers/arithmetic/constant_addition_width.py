# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 AddK.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Bind AddK's allocated constant word and shared Add child to width queries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.resource_generalization import BookkeepingCall, ResourceCall
from ..._component_trace.symbolic_math import SymbolicInt
from ..._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
    schedule_width_steps,
)
from ..control.single_zero_control_width import single_zero_control_width
from .arithmetic_declarations import arithmetic_callees


def constant_addition_width(
    call: ArithmeticCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Allocate, load, add, unload and free a big-endian constant word.

    Target and optional control remain caller-owned. Controlled load/unload
    thread the control around Add. Add owns carries and XorK owns its views.
    Both directions retain the same load constant. Non-Python literals retain
    the source fallback, including its possible workspace underestimate.
    """
    if call.operation not in {"constant_add", "controlled_constant_add"}:
        raise NotImplementedError("No AddK width decomposition for this operation")
    width = call.widths[0]
    controlled = call.operation == "controlled_constant_add"
    if controlled and call.control_value == 0:
        return single_zero_control_width(width, replace(call, control_value=1), child)
    if width == 0:
        raise ValueError("AddK width requires a positive word")
    if type(width) is not int or type(call.constant) is not int:
        return None
    load, addition = arithmetic_callees(call)
    keys = (
        BookkeepingCall("allocate", width),
        load,
        addition,
        load,
        BookkeepingCall("free", width),
    )
    steps = [
        WidthStep("input"),
        WidthStep("allocate"),
        WidthStep("component"),
        WidthStep("component"),
        WidthStep("component"),
        WidthStep("free"),
        WidthStep("output"),
    ]
    connections = [
        WidthConnection(0, 3, width),
        WidthConnection(1, 2, width),
        WidthConnection(2, 3, width),
        WidthConnection(3, 4, width),
        WidthConnection(4, 5, width),
        WidthConnection(3, 6, width),
    ]
    if controlled:
        connections.extend(
            [
                WidthConnection(0, 2, 1),
                WidthConnection(2, 4, 1),
                WidthConnection(4, 6, 1),
            ]
        )
    for group in schedule_width_steps(steps, connections):
        for index in group:
            if 0 < index < 6:
                steps[index] = WidthStep(steps[index].kind, child(keys[index - 1]))
    return estimate_connection_width(steps, connections)
