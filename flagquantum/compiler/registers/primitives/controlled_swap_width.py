# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CSwap.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py

"""Track exact controlled register exchange through native width connections."""

from __future__ import annotations

from collections.abc import Callable

from ..._component_trace.resource_generalization import (
    BookkeepingCall,
    ControlledSwapCall,
    FixedTCall,
    ResourceCall,
)
from ..._component_trace.symbolic_math import SymbolicInt
from ..._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)


def controlled_swap_width(
    call: ControlledSwapCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Track split words, sequential Fredkins and joined output words.

    The control flows through every pair in register order. Splitting and
    joining preserve bit order. This decomposition owns no workspace.
    """
    if call.approximate:
        raise NotImplementedError(
            "No native logical width binding for approximate swap without fanout adjoint"
        )
    width = call.width
    if type(width) is not int:
        return None
    if width < 1:
        raise ValueError("Exact controlled swap width requires nonempty registers")
    split = child(BookkeepingCall("split", width))
    fredkin = child(FixedTCall("fredkin"))
    join = child(BookkeepingCall("join", width))
    steps = [
        WidthStep("input"),
        WidthStep("component", split),
        WidthStep("component", split),
    ]
    steps.extend(WidthStep("component", fredkin) for _ in range(width))
    left_join = len(steps)
    steps.extend(
        (
            WidthStep("component", join),
            WidthStep("component", join),
            WidthStep("output"),
        )
    )
    output = len(steps) - 1
    edges = [WidthConnection(0, 1, width), WidthConnection(0, 2, width)]
    control = 0
    for stage in range(3, 3 + width):
        edges.extend(
            (
                WidthConnection(control, stage, 1),
                WidthConnection(1, stage, 1),
                WidthConnection(2, stage, 1),
                WidthConnection(stage, left_join, 1),
                WidthConnection(stage, left_join + 1, 1),
            )
        )
        control = stage
    edges.extend(
        (
            WidthConnection(control, output, 1),
            WidthConnection(left_join, output, width),
            WidthConnection(left_join + 1, output, width),
        )
    )
    return estimate_connection_width(steps, edges)
