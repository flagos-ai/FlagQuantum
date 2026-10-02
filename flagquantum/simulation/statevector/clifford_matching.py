"""Compilation of exact disjoint Clifford matching layers."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from ...core.ir import Instruction
from ...core.operator_schema import canonical_opcode
from ..native_cpu.permutation import native_cpu_clifford_matching_available
from .fixed_layer_cpu import _matrix_tensors
from .program import (
    _StatevectorCliffordMatchingStep,
    _StatevectorGateStep,
    _StatevectorPreCXStep,
)

_CLIFFORD_PHASE_MAP_CACHE_BYTES = 128 * 1024 * 1024
_CLIFFORD_PHASE_MAP_CACHE: dict[
    tuple[
        int,
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, int], ...],
        str,
        torch.dtype,
    ],
    torch.Tensor,
] = {}


def _clifford_phase_map_cache_bytes() -> int:
    return sum(
        int(mapping.numel()) * int(mapping.element_size())
        for mapping in _CLIFFORD_PHASE_MAP_CACHE.values()
    )


def _store_clifford_phase_map(
    key: tuple[
        int,
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, int], ...],
        str,
        torch.dtype,
    ],
    mapping: torch.Tensor,
) -> torch.Tensor:
    _CLIFFORD_PHASE_MAP_CACHE[key] = mapping
    while (
        len(_CLIFFORD_PHASE_MAP_CACHE) > 1
        and _clifford_phase_map_cache_bytes() > _CLIFFORD_PHASE_MAP_CACHE_BYTES
    ):
        oldest = next(iter(_CLIFFORD_PHASE_MAP_CACHE))
        del _CLIFFORD_PHASE_MAP_CACHE[oldest]
    return mapping


def _encode_clifford_phase_mapping(
    cx_mapping: torch.Tensor,
    step: _StatevectorCliffordMatchingStep,
    n_wires: int,
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...] | None]:
    """Encode each destination's CZ sign into a cached full CX mapping."""

    enabled = (
        bool(step.cz_edges)
        and cx_mapping.numel() == 1 << n_wires
        and os.getenv("FQ_CPU_NATIVE_CLIFFORD_PHASE_MAP", "1").strip().lower()
        not in {"0", "false", "off", "no"}
    )
    if not enabled:
        return cx_mapping, step.cz_edges
    key = (
        n_wires,
        step.controls,
        step.targets,
        step.cz_edges,
        str(cx_mapping.device),
        cx_mapping.dtype,
    )
    cached = _CLIFFORD_PHASE_MAP_CACHE.get(key)
    if cached is not None:
        return cached, None

    destinations = torch.arange(
        cx_mapping.numel(), dtype=torch.int64, device=cx_mapping.device
    )
    negative = torch.zeros(
        cx_mapping.numel(), dtype=torch.bool, device=cx_mapping.device
    )
    for left, right in step.cz_edges:
        left_mask = 1 << (n_wires - left - 1)
        right_mask = 1 << (n_wires - right - 1)
        negative.logical_xor_(
            ((destinations & left_mask) != 0) & ((destinations & right_mask) != 0)
        )
    signed = torch.where(negative, -cx_mapping - 1, cx_mapping).contiguous()
    return _store_clifford_phase_map(key, signed), None


def native_clifford_matching_compile_enabled(
    instructions: Sequence[Instruction],
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    state: torch.Tensor,
    *,
    batch_size: int,
) -> bool:
    """Whether native disjoint CX/CZ matchings are safe and enabled."""

    requires_grad = bool(
        state.requires_grad
        or (
            parameter_bindings is not None
            and any(value.requires_grad for value in parameter_bindings)
        )
        or any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for instruction in instructions
            for value in instruction.params.values()
        )
        or any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for instruction in instructions
            for value in _matrix_tensors(instruction.matrix)
        )
    )
    return bool(
        state.device.type == "cpu"
        and batch_size >= 2
        and not requires_grad
        and native_cpu_clifford_matching_available()
        and os.getenv("FQ_CPU_NATIVE_CLIFFORD_MATCHING", "1").strip().lower()
        not in {"0", "false", "off", "no"}
    )


def fuse_native_disjoint_clifford_matchings(
    program: Sequence[_StatevectorPreCXStep],
) -> list[_StatevectorPreCXStep]:
    """Compile each disjoint matching containing CX into one native step."""

    optimized: list[_StatevectorPreCXStep] = []
    index = 0
    while index < len(program):
        step = program[index]
        if not _is_exact_cx_or_cz(step):
            optimized.append(step)
            index += 1
            continue

        matching: list[_StatevectorGateStep] = []
        occupied_wires: set[int] = set()
        cursor = index
        while cursor < len(program):
            candidate = program[cursor]
            if not _is_exact_cx_or_cz(candidate):
                break
            assert isinstance(candidate, _StatevectorGateStep)
            wires = set(map(int, candidate.instruction.wires))
            if not occupied_wires.isdisjoint(wires):
                break
            matching.append(candidate)
            occupied_wires.update(wires)
            cursor += 1

        cx_steps = tuple(
            item for item in matching if canonical_opcode(item.instruction.name) == "cx"
        )
        cz_edges = tuple(
            (int(item.instruction.wires[0]), int(item.instruction.wires[1]))
            for item in matching
            if canonical_opcode(item.instruction.name) == "cz"
        )
        if len(matching) >= 2 and cx_steps:
            optimized.append(
                _StatevectorCliffordMatchingStep(
                    controls=tuple(int(item.instruction.wires[0]) for item in cx_steps),
                    targets=tuple(int(item.instruction.wires[1]) for item in cx_steps),
                    cz_edges=cz_edges,
                )
            )
        else:
            optimized.extend(matching)
        index = cursor
    return optimized


def _reorder_disjoint_clifford_matchings(
    program: Sequence[_StatevectorPreCXStep],
) -> list[_StatevectorPreCXStep]:
    """Place disjoint CZ edges before CX edges so each kind can be batched."""

    optimized: list[_StatevectorPreCXStep] = []
    index = 0
    while index < len(program):
        step = program[index]
        if not _is_exact_cx_or_cz(step):
            optimized.append(step)
            index += 1
            continue

        matching: list[_StatevectorGateStep] = []
        occupied_wires: set[int] = set()
        cursor = index
        while cursor < len(program):
            candidate = program[cursor]
            if not _is_exact_cx_or_cz(candidate):
                break
            assert isinstance(candidate, _StatevectorGateStep)
            wires = set(map(int, candidate.instruction.wires))
            if not occupied_wires.isdisjoint(wires):
                break
            matching.append(candidate)
            occupied_wires.update(wires)
            cursor += 1

        cz_steps = [
            item for item in matching if canonical_opcode(item.instruction.name) == "cz"
        ]
        cx_steps = [
            item for item in matching if canonical_opcode(item.instruction.name) == "cx"
        ]
        optimized.extend((*cz_steps, *cx_steps) if cz_steps and cx_steps else matching)
        index = cursor
    return optimized


def _is_exact_cx_or_cz(step: _StatevectorPreCXStep) -> bool:
    if not isinstance(step, _StatevectorGateStep):
        return False
    instruction = step.instruction
    return (
        instruction.matrix is None
        and not instruction.params
        and canonical_opcode(instruction.name) in {"cx", "cz"}
    )
