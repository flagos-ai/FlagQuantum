"""Compilation and CPU execution for exact consecutive CZ graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...core.operator_schema import canonical_opcode
from .program import (
    _StatevectorCZGraphStep,
    _StatevectorGateStep,
    _StatevectorPreCXStep,
)


def _fuse_cz_graphs(
    program: Sequence[_StatevectorPreCXStep],
) -> list[_StatevectorPreCXStep]:
    """Replace consecutive exact CZ gates with one parity-phase graph step."""

    optimized: list[_StatevectorPreCXStep] = []
    index = 0
    while index < len(program):
        step = program[index]
        if not _is_exact_cz(step):
            optimized.append(step)
            index += 1
            continue

        cz_steps: list[_StatevectorGateStep] = []
        cursor = index
        while cursor < len(program):
            candidate = program[cursor]
            if not _is_exact_cz(candidate):
                break
            assert isinstance(candidate, _StatevectorGateStep)
            cz_steps.append(candidate)
            cursor += 1

        if len(cz_steps) < 2:
            optimized.extend(cz_steps)
        else:
            edge_parity: dict[tuple[int, int], bool] = {}
            for cz_step in cz_steps:
                left, right = map(int, cz_step.instruction.wires)
                edge = (min(left, right), max(left, right))
                edge_parity[edge] = not edge_parity.get(edge, False)
            edges = tuple(edge for edge, odd in edge_parity.items() if odd)
            if edges:
                optimized.append(_StatevectorCZGraphStep(edges))
        index = cursor
    return optimized


def _is_exact_cz(step: _StatevectorPreCXStep) -> bool:
    return (
        isinstance(step, _StatevectorGateStep)
        and step.instruction.matrix is None
        and not step.instruction.params
        and canonical_opcode(step.instruction.name) == "cz"
    )


def _cz_graph_signs_cpu(
    edges: Sequence[tuple[int, int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Build the compact exact signs for a static graph of CZ gates."""

    graph_wires = tuple(sorted({wire for edge in edges for wire in edge}))
    positions = {wire: index for index, wire in enumerate(graph_wires)}
    previous_neighbor_masks = [0] * len(graph_wires)
    for left, right in edges:
        left_position = positions[int(left)]
        right_position = positions[int(right)]
        earlier, later = sorted((left_position, right_position))
        previous_neighbor_masks[later] ^= 1 << (later - 1 - earlier)

    signs = torch.ones(1, dtype=torch.int8, device=device)
    for position, neighbor_mask in enumerate(previous_neighbor_masks):
        if neighbor_mask:
            modifier = torch.ones(1, dtype=torch.int8, device=device)
            for earlier in range(position):
                connected = neighbor_mask & (1 << (position - 1 - earlier))
                second = -modifier if connected else modifier
                modifier = torch.stack((modifier, second), dim=1).reshape(-1)
        else:
            modifier = torch.ones(1 << position, dtype=torch.int8, device=device)
        signs = torch.stack((signs, signs * modifier), dim=1).reshape(-1)
    return signs, graph_wires


def _apply_cz_graph_cpu(
    state: torch.Tensor,
    signs: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply cached CZ-graph signs with one pass over a CPU statevector."""

    normalized_wires = tuple(int(wire) for wire in wires)
    if signs.dtype != torch.int8 or signs.ndim != 1:
        raise ValueError("CZ graph signs must be a one-dimensional int8 tensor")
    if signs.numel() != 2 ** len(normalized_wires):
        raise ValueError("CZ graph signs do not match the graph wire count")
    if len(set(normalized_wires)) != len(normalized_wires):
        raise ValueError("CZ graph wires must be unique")
    if any(not 0 <= wire < int(n_wires) for wire in normalized_wires):
        raise ValueError("wire is outside the statevector")

    factor_shape = [1] + [1] * int(n_wires)
    for wire in normalized_wires:
        factor_shape[wire + 1] = 2
    tensor = state.reshape((state.shape[0],) + (2,) * int(n_wires))
    return (tensor * signs.reshape(factor_shape)).reshape(state.shape)
