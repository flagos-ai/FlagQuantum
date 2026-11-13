# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 ControlledViaAnd's single zero control.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py

"""Keep a caller control and word live across X/positive-control/X connections."""

from collections.abc import Callable

from ..._component_trace.resource_generalization import CliffordCall, ResourceCall
from ..._component_trace.symbolic_math import SymbolicInt
from ..._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
    schedule_width_steps,
)


def single_zero_control_width(
    width: SymbolicInt,
    positive: ResourceCall,
    child: Callable[[ResourceCall], SymbolicInt],
) -> SymbolicInt:
    """Query the positive child between two control flips, retaining word idling."""
    keys = (CliffordCall("x", 1), positive, CliffordCall("x", 1))
    steps = [
        WidthStep("input"),
        *(WidthStep("component") for _ in keys),
        WidthStep("output"),
    ]
    connections = [
        WidthConnection(0, 1, 1),
        WidthConnection(0, 2, width),
        WidthConnection(1, 2, 1),
        WidthConnection(2, 3, 1),
        WidthConnection(2, 4, width),
        WidthConnection(3, 4, 1),
    ]
    for group in schedule_width_steps(steps, connections):
        for index in group:
            if 0 < index < 4:
                steps[index] = WidthStep("component", child(keys[index - 1]))
    return estimate_connection_width(steps, connections)
