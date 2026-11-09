# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Layout adapted from Qualtran v0.7.0 musical_score.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score.py

"""Lay out register dependencies and storage lifetimes for a musical score."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from heapq import heappop, heappush

from ..compiler._construction.register_expansion import expand_register_calls
from ..compiler._construction.register_profile import (
    register_width,
    verify_register_program,
)
from ..compiler._hybrid.model import HybridProgram, ValueId
from ..compiler.registers.control.multi_control import multi_and_stages


@dataclass(frozen=True)
class RegisterStep:
    """One operation with ordered input and output register identities.

    A register appearing on both sides keeps its lane. Output-only registers
    obtain lanes before input-only lanes are released.
    """

    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    gate: str | None = None
    control_values: tuple[int, int] | None = None


@dataclass(frozen=True)
class RegisterColumn:
    """Placement of an operation in source-like sequential and parallel columns."""

    step: RegisterStep
    sequence: int
    generation: int
    lanes: tuple[int, ...]
    ordinal: int


@dataclass(frozen=True)
class RegisterLine:
    """A width-bearing register lane over an inclusive sequential interval."""

    register: str
    width: int
    lane: int
    start: int
    end: int


@dataclass(frozen=True)
class RegisterScore:
    """Operation placement plus horizontal register lifetimes."""

    columns: tuple[RegisterColumn, ...]
    lines: tuple[RegisterLine, ...]
    output_generation: int


def layout_register_score(
    widths: Mapping[str, int], inputs: Sequence[str], steps: Sequence[RegisterStep]
) -> RegisterScore:
    """Lay out a register graph, with initial registers at sequential column -1.

    Registers still live after all steps end at the output boundary. Register
    identities are single-use across lifetimes. Independent operations share
    a dependency generation; lanes are assigned in sequential display order.
    """
    if any(type(width) is not int or width < 1 for width in widths.values()):
        raise ValueError("Register widths must be positive integers")
    active = dict.fromkeys(inputs, -1)
    if len(active) != len(inputs) or active.keys() - widths.keys():
        raise ValueError("Initial register identities must be unique and declared")
    seen = set(active)
    dependencies: dict[int, tuple[int, ...]] = {}
    if inputs:
        dependencies[-1] = ()
    for index, step in enumerate(steps):
        sources = set(step.inputs)
        targets = set(step.outputs)
        opened = targets - sources
        if len(sources) != len(step.inputs) or len(targets) != len(step.outputs):
            raise ValueError("An operation side contains a duplicate register")
        if sources - active.keys() or opened & seen or opened - widths.keys():
            raise ValueError("Register operation uses an unavailable identity")
        dependencies[index] = tuple(dict.fromkeys(active[key] for key in step.inputs))
        for key in sources - targets:
            del active[key]
        active.update(dict.fromkeys(step.outputs, index))
        seen.update(opened)

    generations: dict[int, int] = {}
    for index in TopologicalSorter(dependencies).static_order():
        generations[index] = 1 + max(
            (generations[pred] for pred in dependencies[index]), default=-1
        )
    output_generation = max(
        max(generations.values(), default=0),
        1 + max((generations[index] for index in active.values()), default=-1),
    )

    live = {key: (lane, -1) for lane, key in enumerate(inputs)}
    available: list[int] = []
    lane_count = len(live)
    columns = []
    lines = []
    for sequence, index in enumerate(key for key in generations if key >= 0):
        step = steps[index]
        for key in step.outputs:
            if key in live:
                continue
            if available:
                lane = heappop(available)
            else:
                lane = lane_count
                lane_count += 1
            live[key] = (lane, sequence)
        lanes = tuple(
            dict.fromkeys(live[key][0] for key in (*step.inputs, *step.outputs))
        )
        columns.append(RegisterColumn(step, sequence, generations[index], lanes, index))
        for key in step.inputs:
            if key not in step.outputs:
                lane, start = live.pop(key)
                lines.append(RegisterLine(key, widths[key], lane, start, sequence))
                heappush(available, lane)

    lines.extend(
        RegisterLine(key, widths[key], lane, start, len(steps))
        for key, (lane, start) in live.items()
    )
    lines.sort(key=lambda line: (line.lane, line.start))
    return RegisterScore(tuple(columns), tuple(lines), output_generation)


def multi_and_register_score(values: tuple[int, ...]) -> RegisterScore:
    """Draw the shared MultiAnd ladder with its live prefix outputs."""
    controls = tuple(f"ctrl[{i}]" for i in range(len(values)))
    prefixes = tuple(f"junk[{i}]" for i in range(len(values) - 2))
    names = (*controls, *prefixes, "target")
    steps = tuple(
        RegisterStep(
            "And",
            (names[left], names[right]),
            (names[left], names[right], names[target]),
            gate="and",
            control_values=cvs,
        )
        for left, right, target, cvs in multi_and_stages(values)
    )
    return layout_register_score(dict.fromkeys(names, 1), controls, steps)


def register_program_score(
    program: HybridProgram, functions: Sequence[HybridProgram] = ()
) -> RegisterScore:
    """Project a verified register construction into display dependencies.

    Root storage is reserved at entry, and locals have explicit declarations
    and releases. The global effect order remains part of execution. The
    displayed generations express register dependencies, including borrowed
    inputs of an implicit uncompute.
    """
    verify_register_program(program, functions)
    expanded = expand_register_calls(program, functions)
    names: dict[ValueId, str] = {}
    widths: dict[str, int] = {}
    inputs: list[str] = []
    steps: list[RegisterStep] = []
    borrowed: dict[ValueId, tuple[str, ...]] = {}
    for op in expanded.body.blocks[0].operations:
        if op.name in {"qreg.declare", "qreg.local"}:
            register = op.results[0]
            name = str(op.attributes["name"])
            names[register.id] = name
            widths[name] = register_width(register, op)
            if op.name == "qreg.declare":
                inputs.append(name)
            else:
                steps.append(RegisterStep(op.name, outputs=(name,)))
            continue
        if op.name == "qreg.return":
            break
        operands = tuple(dict.fromkeys(names[value.id] for value in op.operands[1:]))
        if op.name == "qreg.compute":
            borrowed[op.operands[1].id] = operands[1:]
        elif op.name == "qreg.uncompute":
            operands += borrowed.pop(op.operands[1].id)
        outputs = () if op.name == "qreg.release" else operands
        gate = op.attributes.get("gate")
        steps.append(RegisterStep(op.name, operands, outputs, gate))
    return layout_register_score(widths, inputs, steps)
