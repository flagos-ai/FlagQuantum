"""Dense CPU statevector fusion helpers kept outside the main executor."""

from __future__ import annotations

import os
from collections.abc import Sequence
from math import isfinite
from numbers import Real

import torch

from ...core.operator_schema import canonical_opcode
from ..native_cpu.permutation import (
    compact_cx_rzz_swap_images,
    fused_compact_cx_rzz_swap_out,
)
from .program import (
    _StatevectorCXSequenceRZZSwapStep,
    _StatevectorCXSequenceStep,
    _StatevectorGateStep,
    _StatevectorProgramStep,
    _StatevectorSwapSequenceStep,
)


def _cpu_swap_sequence_fusion_enabled() -> bool:
    return os.getenv("FQ_CPU_SWAP_SEQUENCE_FUSION", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _cpu_cx_rzz_swap_fusion_enabled() -> bool:
    return os.getenv("FQ_CPU_CX_RZZ_SWAP_FUSION", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _fuse_swap_sequences(
    program: Sequence[_StatevectorProgramStep],
    *,
    bounds: tuple[int, int | None] = (2, None),
) -> list[_StatevectorProgramStep]:
    """Combine adjacent SWAPs so their axis permutation is materialized once."""

    minimum_length, maximum_length = bounds
    if minimum_length < 2:
        raise ValueError("SWAP fusion requires a minimum length of at least two")
    if maximum_length is not None and maximum_length < minimum_length:
        raise ValueError("SWAP fusion maximum length must cover the minimum")

    optimized: list[_StatevectorProgramStep] = []
    index = 0
    while index < len(program):
        step = program[index]
        if not (
            isinstance(step, _StatevectorGateStep)
            and canonical_opcode(step.instruction.name) == "swap"
        ):
            optimized.append(step)
            index += 1
            continue
        swaps: list[tuple[int, int]] = []
        cursor = index
        while cursor < len(program):
            candidate = program[cursor]
            if not (
                isinstance(candidate, _StatevectorGateStep)
                and canonical_opcode(candidate.instruction.name) == "swap"
            ):
                break
            left, right = candidate.instruction.wires
            swaps.append((int(left), int(right)))
            cursor += 1
        if len(swaps) >= minimum_length and (
            maximum_length is None or len(swaps) <= maximum_length
        ):
            optimized.append(_StatevectorSwapSequenceStep(tuple(swaps)))
        else:
            optimized.extend(program[index:cursor])
        index = cursor
    return optimized


def _fuse_cx_rzz_swap_sequences(
    program: Sequence[_StatevectorProgramStep],
) -> list[_StatevectorProgramStep]:
    """Fuse static CX-chain, disjoint RZZ, and SWAP triples."""

    optimized: list[_StatevectorProgramStep] = []
    index = 0
    while index < len(program):
        window = program[index : index + 3]
        if len(window) == 3:
            cx_step, rzz_step, swap_step = window
            if (
                isinstance(cx_step, _StatevectorCXSequenceStep)
                and isinstance(rzz_step, _StatevectorGateStep)
                and canonical_opcode(rzz_step.instruction.name) == "rzz"
                and rzz_step.instruction.matrix is None
                and isinstance(swap_step, _StatevectorGateStep)
                and canonical_opcode(swap_step.instruction.name) == "swap"
                and swap_step.instruction.matrix is None
            ):
                angle = rzz_step.instruction.params.get("theta")
                rzz_qubits = tuple(int(qubit) for qubit in rzz_step.instruction.wires)
                swap_qubits = tuple(int(qubit) for qubit in swap_step.instruction.wires)
                if (
                    not isinstance(angle, bool)
                    and isinstance(angle, Real)
                    and isfinite(float(angle))
                    and len(rzz_qubits) == 2
                    and len(swap_qubits) == 2
                    and set(rzz_qubits).isdisjoint(swap_qubits)
                ):
                    optimized.append(
                        _StatevectorCXSequenceRZZSwapStep(
                            controls=cx_step.controls,
                            targets=cx_step.targets,
                            rzz_wires=(rzz_qubits[0], rzz_qubits[1]),
                            rzz_angle=float(angle),
                            swap_wires=(swap_qubits[0], swap_qubits[1]),
                        )
                    )
                    index += 3
                    continue
        optimized.append(program[index])
        index += 1
    return optimized


def _apply_swap_sequence(
    state: torch.Tensor,
    swaps: Sequence[tuple[int, int]],
    n_qubits: int,
) -> torch.Tensor:
    """Apply a SWAP sequence as one final contiguous statevector copy."""

    tensor = state.reshape((state.shape[0],) + (2,) * n_qubits)
    for left, right in swaps:
        tensor = tensor.transpose(int(left) + 1, int(right) + 1)
    return tensor.reshape(state.shape)


def _apply_cx_rzz_swap_sequence(
    step: _StatevectorCXSequenceRZZSwapStep,
    state: torch.Tensor,
    n_qubits: int,
    *,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one compiled static CX/RZZ/SWAP segment natively."""

    images = compact_cx_rzz_swap_images(
        step.controls, step.targets, step.swap_wires, n_qubits
    )
    output = scratch if scratch is not None else torch.empty_like(state)
    if not fused_compact_cx_rzz_swap_out(
        state,
        images,
        output,
        rzz_qubits=step.rzz_wires,
        rzz_angle=step.rzz_angle,
    ):
        raise RuntimeError("compiled CPU CX/RZZ/SWAP kernel became unavailable")
    return output
