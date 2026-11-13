# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute, Adjoint and Cirq import.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/adjoint.py

"""Bind Hamming logical output lifetimes to native resource width queries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

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
from .bit_compression_structure import compression_components
from .hamming_weight import hamming_weight_requirements
from .hamming_weight_structure import hamming_weight_stages


def hamming_weight_width(
    call: ArithmeticCall, child: Callable[[ResourceCall], SymbolicInt]
) -> SymbolicInt | None:
    """Create logical outputs lazily, or consume the supplied computed triple.

    Generic adjoint reverses the forward greedy traversal before assigning its
    instance order. Python zero/one retain source empty-register failures.
    Symbolic inputs use the existing source signature/callee fallback.
    """
    if call.operation != "hamming_weight":
        raise NotImplementedError("No Hamming width for this operation")
    width = call.widths[0]
    if type(width) is not int:
        return None
    if width < 2:
        raise ValueError("Source Hamming width requires at least two input bits")
    junk_width, out_width = hamming_weight_requirements(width)
    source = tuple(range(width))
    junk = tuple(range(width, width + junk_width))
    output = tuple(range(width + junk_width, width + junk_width + out_width))
    producers = dict.fromkeys(source, 0)
    pending_view = True
    steps = [WidthStep("input")]
    connections: list[WidthConnection] = []
    keys: dict[int, ResourceCall] = {}

    def add(
        key: ResourceCall, inputs: tuple[int, ...], outputs: tuple[int, ...]
    ) -> None:
        index = len(steps)
        for wire in inputs:
            connections.append(WidthConnection(producers.pop(wire), index, 1))
        producers.update(dict.fromkeys(outputs, index))
        kind = (
            key.opcode
            if isinstance(key, BookkeepingCall) and key.opcode == "allocate"
            else "component"
        )
        steps.append(WidthStep(kind))
        keys[index] = key

    for stage, wires in hamming_weight_stages(source, junk, output):
        components = (
            compression_components(*wires) if stage == "compress" else (("cx", wires),)
        )
        for kind, operands in components:
            inputs = operands[:2]
            for wire in inputs:
                if pending_view and wire in source:
                    add(BookkeepingCall("split", width), source, source)
                    pending_view = False
                if wire not in producers:
                    add(BookkeepingCall("allocate", 1), (), (wire,))
            if kind == "cx":
                add(CliffordCall("cx", 2), operands, operands)
            else:
                add(ConjunctionCall((1, 1)), inputs, operands)
    for word in (source, junk, output):
        if len(word) > 1:
            add(BookkeepingCall("join", len(word)), word, word)
    end = len(steps)
    connections.extend(WidthConnection(index, end, 1) for index in producers.values())
    steps.append(WidthStep("output"))
    if call.adjoint:
        order = tuple(reversed(schedule_width_steps(steps, connections)[0]))
        indices = {old: new for new, old in enumerate(order)}
        inverse_kinds: dict[str, Literal["input", "output", "free", "component"]] = {
            "input": "output",
            "output": "input",
            "allocate": "free",
        }
        steps = [
            replace(
                steps[index], kind=inverse_kinds.get(steps[index].kind, "component")
            )
            for index in order
        ]
        connections = [
            WidthConnection(indices[edge.target], indices[edge.source], edge.qubits)
            for edge in connections
        ]
        keys = {indices[index]: _adjoint_key(key) for index, key in keys.items()}
    for group in schedule_width_steps(steps, connections):
        for index in group:
            if index in keys:
                steps[index] = replace(steps[index], qubits=child(keys[index]))
    return estimate_connection_width(steps, connections)


def _adjoint_key(key: ResourceCall) -> ResourceCall:
    if isinstance(key, BookkeepingCall):
        inverse: dict[str, Literal["split", "join", "free"]] = {
            "split": "join",
            "join": "split",
            "allocate": "free",
        }
        return replace(
            key,
            opcode=inverse[key.opcode],
        )
    if isinstance(key, ConjunctionCall):
        return replace(key, adjoint=True)
    return key
