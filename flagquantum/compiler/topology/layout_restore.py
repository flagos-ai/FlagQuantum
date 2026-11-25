"""Restore logical layouts through legal coupling-map SWAPs."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from .coupling import CouplingMap

if TYPE_CHECKING:
    from .sabre import SabrePlan


def _restore_path(
    start: int,
    goal: int,
    tree: list[list[int]],
    active: list[bool],
) -> list[int]:
    """Return the remaining tree path from ``start`` to ``goal``, inclusive."""

    previous = {start: -1}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for neighbour in tree[node]:
            if active[neighbour] and neighbour not in previous:
                previous[neighbour] = node
                queue.append(neighbour)
    if goal not in previous:
        raise ValueError(
            f"layout restore has no coupling path from wire {start} to wire {goal}"
        )
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def restore_swaps(
    layout: tuple[int, ...],
    coupling: CouplingMap,
) -> tuple[tuple[int, int], ...]:
    """Plan the SWAPs that move every logical qubit of ``layout`` to its own qubit.

    Replaying a forward plan in reverse only returns the layout that plan started
    from, so a plan that started away from the identity layout needs an explicit
    restore. The coupling graph is reduced to a spanning forest and one tree leaf
    at a time is walked to its home qubit along the edges that are still active. A
    leaf has at most one active edge, so a qubit that has already reached home is
    never displaced.

    Args:
        layout: Physical qubit for each logical qubit.
        coupling: Topology the restore SWAPs must respect.

    Raises:
        ValueError: If ``layout`` is not a permutation of the circuit qubits, or if
            the coupling map has no spanning forest over them.
    """

    placement = tuple(layout)
    n_wires = len(placement)
    if sorted(placement) != list(range(n_wires)):
        raise ValueError(
            f"layout must be a permutation of the circuit wires, got {placement!r}"
        )

    adjacency: list[list[int]] = [[] for _ in range(n_wires)]
    for left, right in coupling.edges:
        if left < n_wires and right < n_wires:
            adjacency[left].append(right)
            adjacency[right].append(left)
    # A layout already at home needs no spanning-forest traversal. Keep input
    # validation and edge accounting above this fast path unchanged.
    if placement == tuple(range(n_wires)):
        return ()

    parent = [-1] * n_wires
    for root in range(n_wires):
        if parent[root] != -1:
            continue
        parent[root] = root
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbour in sorted(adjacency[node]):
                if parent[neighbour] == -1:
                    parent[neighbour] = node
                    queue.append(neighbour)
    tree = [
        [
            neighbour
            for neighbour in adjacency[node]
            if parent[neighbour] == node or parent[node] == neighbour
        ]
        for node in range(n_wires)
    ]

    logical_to_physical = list(placement)
    physical_to_logical = [0] * n_wires
    for logical, physical in enumerate(logical_to_physical):
        physical_to_logical[physical] = logical
    swaps: list[tuple[int, int]] = []
    active = [True] * n_wires
    for _ in range(n_wires):
        leaf = next(
            (
                node
                for node in range(n_wires)
                if active[node]
                and sum(1 for neighbour in tree[node] if active[neighbour]) <= 1
            ),
            None,
        )
        if leaf is None:
            raise ValueError("coupling map has no spanning forest over the circuit")
        physical = logical_to_physical[leaf]
        # The active tree stays fixed until this token reaches its leaf. Walk
        # its unique path once; SWAPs move tokens, not the remaining tree edges.
        path = _restore_path(physical, leaf, tree, active) if physical != leaf else ()
        for step in path[1:]:
            other = physical_to_logical[step]
            logical_to_physical[leaf] = step
            logical_to_physical[other] = physical
            physical_to_logical[step] = leaf
            physical_to_logical[physical] = other
            swaps.append((physical, step) if physical < step else (step, physical))
            physical = step
        active[leaf] = False
    return tuple(swaps)


def plan_restore_swaps(
    plan: SabrePlan,
    coupling: CouplingMap,
) -> tuple[tuple[int, int, int], ...]:
    """Plan the SWAPs that return a planned program to the identity layout.

    Replaying ``SabrePlan.swaps`` in reverse exactly undoes the forward SWAPs, so
    it returns the layout the plan started from. That is the identity layout only
    for a plan that started there. Two restores are therefore always available:
    walk the plan's final layout home directly, or replay the forward SWAPs in
    reverse and then walk the layout the plan started from home. The shorter one
    is returned, so a plan that started from the identity layout pays no more
    than its forward SWAPs.

    Returns:
        ``(left, right, anchor)`` triples in execution order. ``anchor`` is the
        source instruction index a replayed forward SWAP was attributed to, or
        ``SabrePlan.swap_anchors[0]`` for a swap of the tree walk.

    Raises:
        ValueError: If the coupling map has no spanning forest over the plan's
            qubits.
    """

    replayed = tuple(
        (plan.swaps[index][0], plan.swaps[index][1], plan.swap_anchors[index])
        for index in range(len(plan.swaps) - 1, -1, -1)
    )
    if plan.swaps:
        anchor = plan.swap_anchors[0]
        unwound = plan.initial_logical_to_physical
        candidate = replayed + tuple(
            (left, right, anchor) for left, right in restore_swaps(unwound, coupling)
        )
        direct = tuple(
            (left, right, anchor)
            for left, right in restore_swaps(plan.final_logical_to_physical, coupling)
        )
        if len(direct) < len(candidate):
            return direct
        return candidate
    anchor = plan.swap_anchors[0] if plan.swap_anchors else 0
    return tuple(
        (left, right, anchor)
        for left, right in restore_swaps(plan.final_logical_to_physical, coupling)
    )
