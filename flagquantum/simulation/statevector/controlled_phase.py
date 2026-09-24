"""Compile exact portable controlled-phase decompositions for CPU execution."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import TypeAlias

import torch

from ...core.operator_schema import canonical_opcode
from .program import (
    _StatevectorControlledPhaseDecompositionStep,
    _StatevectorControlledPhaseGraphStep,
    _StatevectorGateStep,
    _StatevectorPreCXStep,
    _StatevectorRXRZLoopStep,
)

_ControlledPhaseStep: TypeAlias = (
    _StatevectorGateStep
    | _StatevectorRXRZLoopStep
    | _StatevectorControlledPhaseDecompositionStep
)


def _constant_rz_angle(step: _StatevectorGateStep) -> float | None:
    """Return a scalar constant RZ angle, declining every ambiguous case."""

    instruction = step.instruction
    if (
        instruction.matrix is not None
        or canonical_opcode(instruction.name) != "rz"
        or len(instruction.wires) != 1
    ):
        return None
    angle = instruction.params.get("theta")
    if isinstance(angle, bool) or not isinstance(angle, Real):
        return None
    real_angle = float(angle)
    return real_angle if isfinite(real_angle) else None


def _fuse_controlled_phase_decompositions(
    program: Sequence[_StatevectorGateStep | _StatevectorRXRZLoopStep],
) -> list[_ControlledPhaseStep]:
    """Fuse the exact ``RZ-RZ-CX-RZ-CX`` controlled-phase decomposition.

    Only literal scalar angles with exact equality are accepted. Parameterized
    and approximately matching sequences deliberately remain on the general
    path so this optimization cannot infer an algebraic relation that the IR
    does not prove.
    """

    optimized: list[_ControlledPhaseStep] = []
    index = 0
    while index < len(program):
        window = program[index : index + 5]
        gate_window = tuple(
            step for step in window if isinstance(step, _StatevectorGateStep)
        )
        if len(gate_window) == 5:
            first, second, first_cx, inverse, second_cx = gate_window
            first_angle = _constant_rz_angle(first)
            second_angle = _constant_rz_angle(second)
            inverse_angle = _constant_rz_angle(inverse)
            first_cx_instruction = first_cx.instruction
            second_cx_instruction = second_cx.instruction
            control = first.instruction.wires[0]
            target = second.instruction.wires[0]
            if (
                control != target
                and first_angle is not None
                and second_angle == first_angle
                and inverse_angle == -first_angle
                and inverse.instruction.wires == (target,)
                and first_cx_instruction.matrix is None
                and second_cx_instruction.matrix is None
                and canonical_opcode(first_cx_instruction.name) == "cx"
                and canonical_opcode(second_cx_instruction.name) == "cx"
                and first_cx_instruction.wires == (control, target)
                and second_cx_instruction.wires == (control, target)
            ):
                optimized.append(
                    _StatevectorControlledPhaseDecompositionStep(
                        control=control,
                        target=target,
                        half_angle=first_angle,
                    )
                )
                index += 5
                continue
        optimized.append(program[index])
        index += 1
    return optimized


def _fuse_controlled_phase_graphs(
    program: Sequence[_StatevectorPreCXStep],
) -> list[_StatevectorPreCXStep]:
    """Combine consecutive static controlled phases into one weighted graph."""

    optimized: list[_StatevectorPreCXStep] = []
    index = 0
    while index < len(program):
        step = program[index]
        if not isinstance(step, _StatevectorControlledPhaseDecompositionStep):
            optimized.append(step)
            index += 1
            continue

        phase_steps: list[_StatevectorControlledPhaseDecompositionStep] = []
        cursor = index
        while cursor < len(program):
            candidate = program[cursor]
            if not isinstance(candidate, _StatevectorControlledPhaseDecompositionStep):
                break
            phase_steps.append(candidate)
            cursor += 1

        if len(phase_steps) < 2:
            optimized.extend(phase_steps)
        else:
            optimized.append(
                _StatevectorControlledPhaseGraphStep(
                    tuple(
                        (step.control, step.target, step.half_angle)
                        for step in phase_steps
                    )
                )
            )
        index = cursor
    return optimized


def _controlled_phase_graph_factors_cpu(
    edges: Sequence[tuple[int, int, float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Build exact amplitude factors for a static controlled-phase graph."""

    if dtype not in {torch.complex64, torch.complex128}:
        raise TypeError("controlled-phase graph requires a complex state dtype")
    graph_wires = tuple(
        sorted(
            {int(wire) for control, target, _ in edges for wire in (control, target)}
        )
    )
    if not graph_wires:
        raise ValueError("controlled-phase graph requires at least one edge")
    if any(int(control) == int(target) for control, target, _ in edges):
        raise ValueError("controlled-phase graph edges require distinct wires")

    positions = {wire: position for position, wire in enumerate(graph_wires)}
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    basis = torch.arange(1 << len(graph_wires), device=device, dtype=torch.int64)
    phases = torch.full(
        basis.shape,
        -sum(float(half_angle) / 2.0 for _, _, half_angle in edges),
        device=device,
        dtype=real_dtype,
    )
    for control, target, half_angle in edges:
        control_shift = len(graph_wires) - 1 - positions[int(control)]
        target_shift = len(graph_wires) - 1 - positions[int(target)]
        selected = ((basis >> control_shift) & 1) * ((basis >> target_shift) & 1)
        phases = phases + selected.to(real_dtype) * (2.0 * float(half_angle))
    factors = torch.polar(torch.ones_like(phases), phases).to(dtype=dtype)
    return factors, graph_wires


def _apply_controlled_phase_graph_cpu(
    state: torch.Tensor,
    factors: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply precomputed controlled-phase graph factors in one state pass."""

    normalized_wires = tuple(int(wire) for wire in wires)
    if factors.ndim != 1 or factors.dtype != state.dtype:
        raise ValueError("controlled-phase graph factors must match the state dtype")
    if factors.numel() != 2 ** len(normalized_wires):
        raise ValueError("controlled-phase graph factors do not match the wire count")
    if len(set(normalized_wires)) != len(normalized_wires):
        raise ValueError("controlled-phase graph wires must be unique")
    if any(not 0 <= wire < int(n_wires) for wire in normalized_wires):
        raise ValueError("wire is outside the statevector")

    factor_shape = [1] + [1] * int(n_wires)
    for wire in normalized_wires:
        factor_shape[wire + 1] = 2
    tensor = state.reshape((state.shape[0],) + (2,) * int(n_wires))
    return (tensor * factors.reshape(factor_shape)).reshape(state.shape)
