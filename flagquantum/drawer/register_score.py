# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Layout adapted from Qualtran v0.7.0 musical_score.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score.py

"""Lay out register dependencies and storage lifetimes for a musical score."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from math import prod
from typing import Literal

from ..compiler._construction.register_expansion import expand_register_calls
from ..compiler._construction.register_profile import (
    register_width,
    verify_register_program,
)
from ..compiler._hybrid.model import HybridProgram, ValueId
from ..compiler.registers.control.multi_control import multi_and_stages
from ..core.ir.program.register_types import opaque_register_type
from ..core.ir.program.registers import Register, RegisterSide
from .register_lanes import RegisterLaneManager, register_line_flavor


@dataclass(frozen=True)
class RegisterStep:
    """One operation with ordered input and output register identities.

    A register appearing on both sides keeps its lane. Output-only registers
    obtain lanes before input-only lanes are released.
    Optional Core metadata retains array grouping for allocation callbacks.
    """

    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    gate: str | None = None
    control_values: tuple[int, int] | None = None
    input_registers: tuple[Register, ...] = ()
    output_registers: tuple[Register, ...] = ()


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
    flavor: Literal["HLineFlavor.QUANTUM", "HLineFlavor.CLASSICAL"] = (
        "HLineFlavor.QUANTUM"
    )


@dataclass(frozen=True)
class RegisterScore:
    """Operation placement, lifetimes and final signature identity order.

    output_order retains the final connection order for symbol callbacks.
    Hand-authored scores may omit it to use completed-line order.
    """

    columns: tuple[RegisterColumn, ...]
    lines: tuple[RegisterLine, ...]
    output_generation: int
    output_order: tuple[str, ...] = ()


def _register_groups(
    keys: Sequence[str],
    registers: Sequence[Register],
    widths: Mapping[str, int],
    through: Sequence[str] = (),
    *,
    side: RegisterSide = RegisterSide.THRU,
) -> tuple[tuple[Register, tuple[str, ...]], ...]:
    """Retain Core array grouping, synthesizing scalar metadata for plain layouts."""
    if not registers:
        return tuple(
            (
                Register(
                    key,
                    opaque_register_type(widths[key]),
                    side=RegisterSide.THRU if key in through else side,
                ),
                (key,),
            )
            for key in keys
        )
    groups = []
    offset = 0
    for register in registers:
        size = prod(register.shape)
        elements = tuple(keys[offset : offset + size])
        if len(elements) != size or any(
            widths[key] != register.bitsize for key in elements
        ):
            raise ValueError(
                "Score register metadata must match element widths and counts"
            )
        if sum(key in through for key in elements) not in (0, size):
            raise ValueError("Score register metadata cannot partially retain an array")
        groups.append((register, elements))
        offset += size
    if offset != len(keys):
        raise ValueError("Score register metadata must cover every element")
    return tuple(groups)


def layout_register_score(
    widths: Mapping[str, int],
    inputs: Sequence[str],
    steps: Sequence[RegisterStep],
    *,
    lane_manager: RegisterLaneManager | None = None,
    input_registers: Sequence[Register] = (),
    output_registers: Sequence[Register] = (),
    outputs: Sequence[str] | None = None,
) -> RegisterScore:
    """Lay out a register graph, with initial registers at sequential column -1.

    Registers still live after all steps end at the output boundary. Register
    identities are single-use across lifetimes. Independent operations share
    a dependency generation; lanes are assigned in sequential display order.
    A caller-owned lane_manager can reserve display lanes. Core register
    metadata preserves per-register predicate and per-element hook cadence.
    The manager's seq_x cursor and partial hlines are current at each hook.
    Plain layouts synthesize scalar opaque metadata from their declared widths.
    Explicit outputs preserve final signature order for release callbacks.
    """
    if any(type(width) is not int or width < 0 for width in widths.values()):
        raise ValueError("Register widths must be nonnegative integers")
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

    initial = _register_groups(inputs, input_registers, widths)
    groups = tuple(
        (
            _register_groups(step.inputs, step.input_registers, widths, step.outputs),
            _register_groups(step.outputs, step.output_registers, widths, step.inputs),
        )
        for step in steps
    )
    final_keys = tuple(active) if outputs is None else tuple(outputs)
    if len(set(final_keys)) != len(final_keys) or set(final_keys) != set(active):
        raise ValueError("Score outputs must retain every live register exactly once")
    final = _register_groups(final_keys, output_registers, widths)
    manager = RegisterLaneManager() if lane_manager is None else lane_manager
    manager.seq_x = -1
    live: dict[str, tuple[int, int]] = {}
    flavors: dict[str, Literal["HLineFlavor.QUANTUM", "HLineFlavor.CLASSICAL"]] = {}
    for register, keys in initial:
        for key, lane in zip(keys, manager.new("input", register), strict=True):
            live[key] = (lane, -1)
            flavors[key] = register_line_flavor(register.dtype)
    columns = []
    lines = []
    for sequence, index in enumerate(key for key in generations if key >= 0):
        manager.seq_x = sequence
        step = steps[index]
        left, right = groups[index]
        for register, keys in right:
            if (keys and keys[0] in live) or (
                not keys and register.side == RegisterSide.THRU
            ):
                continue
            for key, lane in zip(keys, manager.new(index, register), strict=True):
                live[key] = (lane, sequence)
                flavors[key] = register_line_flavor(register.dtype)
        lanes = tuple(
            dict.fromkeys(live[key][0] for key in (*step.inputs, *step.outputs))
        )
        columns.append(RegisterColumn(step, sequence, generations[index], lanes, index))
        for register, keys in left:
            if not set(keys).intersection(step.outputs):
                released = []
                for key in keys:
                    lane, start = live.pop(key)
                    lines.append(
                        RegisterLine(
                            key, widths[key], lane, start, sequence, flavors[key]
                        )
                    )
                    released.append(lane)
                manager.free(index, register, tuple(released))

    manager.seq_x = len(steps)
    for register, keys in final:
        released = []
        for key in keys:
            lane, start = live.pop(key)
            lines.append(
                RegisterLine(key, widths[key], lane, start, len(steps), flavors[key])
            )
            released.append(lane)
        manager.free("output", register, tuple(released))
    lines.sort(key=lambda line: (line.lane, line.start))
    return RegisterScore(tuple(columns), tuple(lines), output_generation, final_keys)


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
