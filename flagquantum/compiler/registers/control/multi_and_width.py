# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiAnd._decompose_via_tree and its adjoint.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/and_bloq.py

"""Build And-ladder width connections using the existing native stages."""

from __future__ import annotations

from collections.abc import Callable

from ..._component_trace.resource_generalization import ConjunctionCall, ResourceCall
from ..._component_trace.symbolic_math import SymbolicInt
from ..._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)
from .multi_control import multi_and_stages


def multi_and_width(
    call: ConjunctionCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Return connected peak for concrete polarities, or defer to fallback.

    Controls are big-endian logical inputs. Each ladder stage creates one
    prefix output retained through the final boundary. The adjoint reverses
    these same connections and consumes prefix outputs.
    """
    values = call.controls
    if not isinstance(values, tuple) or len(values) == 2:
        return None
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise NotImplementedError("And width connections require binary polarities")
    # Validation above admits exactly the native emitter's concrete values.
    concrete = tuple(int(value) for value in values)
    stages = tuple(multi_and_stages(concrete))
    output = len(stages) + 1
    producers = dict.fromkeys(range(len(values)), 0)
    connections = []
    for index, (left, right, target, _) in enumerate(stages, 1):
        for wire in (left, right):
            connections.append(WidthConnection(producers[wire], index, 1))
            producers[wire] = index
        producers[target] = index
    connections.extend(
        WidthConnection(source, output, 1) for source in producers.values()
    )
    calls = tuple(ConjunctionCall(cvs, call.adjoint) for _, _, _, cvs in stages)
    if call.adjoint:
        calls = tuple(reversed(calls))
        connections = [
            WidthConnection(output - edge.target, output - edge.source, edge.qubits)
            for edge in connections
        ]
    steps = [WidthStep("input")]
    steps.extend(WidthStep("component", child(stage)) for stage in calls)
    steps.append(WidthStep("output"))
    return estimate_connection_width(steps, connections)
