# Adapted from rustworkx-core/src/token_swapper.rs at
# 69d412ad9b915d5f86608c15d6277c6d9195ee0a (Apache License, Version 2.0).
# See https://www.apache.org/licenses/LICENSE-2.0 and the repository LICENSE.
# Modified for FlagQuantum: complete native layouts, deterministic traversal,
# bounded incumbent comparison, and no external graph or random trial objects.
"""Bounded happy-swap restoration candidate; not enabled in production."""

from __future__ import annotations

from .coupling import CouplingMap


def _token_swaps(
    layout: tuple[int, ...],
    coupling: CouplingMap,
    max_swaps: int,
) -> tuple[tuple[int, int], ...] | None:
    """Restore a valid complete layout or abstain within the swap budget.

    Each directed edge moves its token one hop closer to home. A directed
    cycle admits a happy swap chain; a path to a settled token admits one
    unhappy swap. The caller retains its existing route unless this complete
    candidate is cheaper. Partial routes never escape on budget exhaustion.
    """
    width = len(layout)
    tokens = [0] * width
    for logical, physical in enumerate(layout):
        tokens[physical] = logical
    adjacency = tuple(
        tuple(neighbor for neighbor in coupling.neighbors(wire) if neighbor < width)
        for wire in range(width)
    )
    # Exclude unused device wires from distances as well as emitted swaps.
    graph = (
        coupling
        if coupling.n_qubits == width
        else CouplingMap(width, (edge for edge in coupling.edges if max(edge) < width))
    )
    distances = graph.distance_matrix()
    if any(distances[physical][logical] < 0 for physical, logical in enumerate(tokens)):
        return None
    swaps: list[tuple[int, int]] = []
    while any(physical != logical for physical, logical in enumerate(tokens)):
        if len(swaps) >= max_swaps:
            return None
        node = next(
            physical for physical, logical in enumerate(tokens) if physical != logical
        )
        path: list[int] = []
        positions: dict[int, int] = {}
        while node not in positions and tokens[node] != node:
            positions[node] = len(path)
            path.append(node)
            home = tokens[node]
            node = next(
                neighbor
                for neighbor in adjacency[node]
                if distances[neighbor][home] < distances[node][home]
            )
        moves: tuple[tuple[int, int], ...]
        if tokens[node] == node:
            moves = ((path[-1], node),)
        else:
            cycle = path[positions[node] :] + [node]
            moves = tuple(zip(cycle[1:-1], cycle[2:], strict=True))[::-1]
        if len(swaps) + len(moves) > max_swaps:
            return None
        for left, right in moves:
            tokens[left], tokens[right] = tokens[right], tokens[left]
            swaps.append((min(left, right), max(left, right)))
    return tuple(swaps)
