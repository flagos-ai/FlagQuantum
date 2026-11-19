# Portions adapted from Cirq 1.7.0, transformers/routing/route_circuit_cqc.py.
# Copyright 2022 The Cirq Developers
# Licensed under the Apache License, Version 2.0; see the repository LICENSE.
# Modified for FlagQuantum: bounded dependency-DAG layers, physical SWAPs,
# immutable layout evaluation and no Cirq or NetworkX runtime dependency.

"""Bounded layer-cost SWAP proposals for the native routing planner."""

from __future__ import annotations

import itertools


def choose_layer_swaps(
    candidates: set[tuple[int, int]],
    front: list[int],
    qubit_pairs: tuple[tuple[int, ...], ...],
    blocks: tuple[bool, ...],
    successors: list[list[int]],
    pending: list[int],
    layout: list[int],
    distances: tuple[tuple[float, ...], ...],
) -> tuple[tuple[int, int], ...] | None:
    """Filter single, then disjoint paired SWAPs through bounded future layers."""
    # Do not copy the full dependency graph or indegree array for every proposal.
    # Count only predecessor edges visited within this bounded projection.
    layers: list[tuple[tuple[int, ...], ...]] = []
    ready = list(front)
    decrements: dict[int, int] = {}
    visited = 0
    while ready and len(layers) < 8 and visited + len(ready) <= 160:
        layer = tuple(qubit_pairs[index] for index in ready if blocks[index])
        if layer:
            layers.append(layer)
        visited += len(ready)
        following = []
        for index in ready:
            for successor in successors[index]:
                decrements[successor] = decrements.get(successor, 0) + 1
                if decrements[successor] == pending[successor]:
                    following.append(successor)
        ready = following
    if not layers:
        return None
    complete = not ready and len(layers) < 8
    ordered = sorted(candidates)

    def select(
        options: list[tuple[tuple[int, int], ...]],
    ) -> tuple[tuple[int, int], ...] | None:
        for layer in layers:
            if len(options) <= 1:
                break
            costs = []
            for moves in options:
                exchanged = {
                    source: destination
                    for left, right in moves
                    for source, destination in ((left, right), (right, left))
                }
                lengths = [
                    distances[exchanged.get(layout[a], layout[a])][
                        exchanged.get(layout[b], layout[b])
                    ]
                    for a, b in layer
                ]
                costs.append((max(lengths), sum(lengths)))
            minimum = min(costs)
            options = [
                option
                for option, cost in zip(options, costs, strict=True)
                if cost == minimum
            ]
        if not options or (len(options) > 1 and not complete):
            return None
        return options[0]

    single = select([(edge,) for edge in ordered])
    if single is not None:
        return single
    # At most 496 disjoint-pair candidates are considered. Above this size,
    # return to the existing single-SWAP heuristic instead of unbounded growth.
    if len(ordered) > 32:
        return None
    return select(
        [
            (first, second)
            for first, second in itertools.combinations(ordered, 2)
            if set(first).isdisjoint(second)
        ]
    )
