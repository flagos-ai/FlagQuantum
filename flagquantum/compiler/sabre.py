"""SABRE-style lookahead SWAP planning for one undirected coupling topology.

The planner implements the SABRE heuristic (Li, Ding, and Xie, "Tackling the
Qubit Mapping Problem for NISQ-Era Quantum Devices", ASPLOS 2019). It keeps a
front layer of two-wire operations whose predecessors have already executed, and
when no front-layer operation is a coupling edge it inserts the SWAP that
minimizes the mean front-layer hop distance plus a distance-decayed lookahead
term over a bounded extended layer.

The plan is one persistent layout for the whole circuit, so a materializer
replays the forward SWAPs in reverse at the end to restore the logical output
layout. The planner is deterministic: ties between equal-cost SWAP candidates
resolve to the lexicographically smallest physical pair, so the same program and
coupling map always produce the same plan without a seed.

Unlike the shortest-path strategies, this planner moves the layout before an
operation rather than immediately before it, which is what makes it effective on
devices of 20 or more wires.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.ir import CircuitIR

if TYPE_CHECKING:
    from .routing import CouplingMap

# SABRE defaults. The extended layer is bounded so planning stays linear in the
# circuit size, and the lookahead weight keeps the front layer dominant.
LOOKAHEAD_DECAY = 0.9
LOOKAHEAD_EXTENDED_SIZE = 20
LOOKAHEAD_WEIGHT = 0.5

# A SWAP that executes no front-layer operation is only useful if it shortens
# some blocked front-layer distance. On a connected topology of N wires a useful
# SWAP always exists within the graph diameter, so this many consecutive
# stalled SWAPs means the front layer cannot be unblocked.
STALLED_SWAP_LIMIT_PER_WIRE = 4


@dataclass(frozen=True)
class SabrePlan:
    """A planned persistent layout and the forward SWAPs that realize it.

    Attributes:
        sequence: Source instruction indices in the order the planner places
            them. A front-layer operation is emitted as soon as its
            predecessors are placed, so this differs from source order when two
            independent operations become executable at different times.
        placements: Physical wires for every source instruction, indexed by
            source instruction index.
        swaps: Forward SWAP pairs in the order the planner applied them.
        swap_placements: Number of instructions already placed when each forward
            SWAP was applied, so materialization can interleave SWAPs with
            instructions.
        swap_anchors: Source instruction index each forward SWAP is attributed
            to, namely the lowest-indexed blocked front-layer operation when the
            SWAP was chosen. The heuristic picks a SWAP for its effect on the
            whole front layer, so this is an audit anchor, not a sole cause.
        final_logical_to_physical: Layout after the forward SWAPs. Replaying
            ``swaps`` in reverse restores the identity layout.
    """

    sequence: tuple[int, ...]
    placements: tuple[tuple[int, ...], ...]
    swaps: tuple[tuple[int, int], ...]
    swap_placements: tuple[int, ...]
    swap_anchors: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]


def _shifted_distance(
    wire_pair: tuple[int, ...],
    left: int,
    right: int,
    logical_to_physical: list[int],
    distances: tuple[tuple[float, ...], ...],
) -> float:
    """Return a two-wire hop distance as it would be after a candidate SWAP."""

    first = logical_to_physical[wire_pair[0]]
    second = logical_to_physical[wire_pair[1]]
    if first == left:
        first = right
    elif first == right:
        first = left
    if second == left:
        second = right
    elif second == right:
        second = left
    return distances[first][second]


def plan_sabre_swaps(program: CircuitIR, coupling: CouplingMap) -> SabrePlan:
    """Plan a SABRE persistent layout for one program on one coupling map.

    Only the wires owned by the program are used, and inserted SWAPs always join
    two wires of the program, so no physical ancilla is required.

    Raises:
        ValueError: If the coupling map has fewer wires than the program, or if
            the front layer cannot be unblocked within the stalled-SWAP limit.
    """

    n_wires = program.n_wires
    if coupling.n_wires < n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")
    instructions = program.instructions
    count = len(instructions)
    identity = tuple(range(n_wires))
    if count == 0:
        return SabrePlan((), (), (), (), (), identity)

    # ``CouplingMap.distance_matrix`` marks a disconnected pair with a negative
    # sentinel. A cost function needs such a pair to be maximally expensive
    # rather than maximally cheap, so a local infinite-distance copy is used.
    distances = tuple(
        tuple(math.inf if hop < 0 else float(hop) for hop in row)
        for row in coupling.distance_matrix()
    )
    neighbours = tuple(coupling.neighbors(wire) for wire in range(coupling.n_wires))
    wires = tuple(instruction.wires for instruction in instructions)
    # A channel occupies its wires but must not be pulled onto a coupling edge,
    # so it never blocks the front layer.
    blocks = tuple(
        len(wire_pair) == 2 and not instruction.metadata.get("is_channel")
        for instruction, wire_pair in zip(instructions, wires, strict=True)
    )
    # A SWAP along one coupling edge keeps each logical wire inside its own
    # connected component, so a two-wire operation whose logical wires have no
    # coupling path between them can never be legalized.
    for index, wire_pair in enumerate(wires):
        if blocks[index] and math.isinf(distances[wire_pair[0]][wire_pair[1]]):
            raise ValueError(
                f"no coupling path between wires {wire_pair[0]} and "
                f"{wire_pair[1]} of two-wire instruction {index}"
            )

    # One edge per (instruction, wire) pair, so an instruction whose two wires
    # share a predecessor still reaches zero pending predecessors.
    successors: list[list[int]] = [[] for _ in range(count)]
    pending = [0] * count
    last_on_wire: dict[int, int] = {}
    for index, wire_pair in enumerate(wires):
        for wire in wire_pair:
            previous = last_on_wire.get(wire)
            if previous is not None:
                successors[previous].append(index)
                pending[index] += 1
            last_on_wire[wire] = index

    logical_to_physical = list(range(n_wires))
    physical_to_logical = list(range(n_wires))
    placements: list[tuple[int, ...]] = [() for _ in range(count)]
    order: list[int] = []
    swaps: list[tuple[int, int]] = []
    swap_placements: list[int] = []
    swap_anchors: list[int] = []
    front = [index for index in range(count) if pending[index] == 0]
    executed = 0
    stalled = 0

    def separation(index: int) -> float:
        left = logical_to_physical[wires[index][0]]
        right = logical_to_physical[wires[index][1]]
        return distances[left][right]

    def extended_layer(blocked_front: list[int]) -> list[int]:
        """Return blocked operations reachable from the front, in BFS order."""

        extended: list[int] = []
        seen = set(front)
        queue: deque[int] = deque()
        for index in blocked_front:
            for successor in successors[index]:
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        while queue and len(extended) < LOOKAHEAD_EXTENDED_SIZE:
            index = queue.popleft()
            if blocks[index]:
                extended.append(index)
            for successor in successors[index]:
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        return extended

    def best_swap(blocked_front: list[int]) -> tuple[int, int] | None:
        """Return the lowest-cost SWAP candidate, or None if none exists."""

        candidates: set[tuple[int, int]] = set()
        for index in blocked_front:
            for wire in wires[index]:
                physical = logical_to_physical[wire]
                for neighbour in neighbours[physical]:
                    candidates.add(
                        (physical, neighbour)
                        if physical < neighbour
                        else (neighbour, physical)
                    )
        if not candidates:
            return None
        extended = extended_layer(blocked_front)
        best: tuple[int, int] | None = None
        best_score = 0.0
        # Sorted iteration makes an equal-cost tie deterministic.
        for left, right in sorted(candidates):
            front_total = 0.0
            for index in blocked_front:
                front_total += _shifted_distance(
                    wires[index], left, right, logical_to_physical, distances
                )
            score = front_total / len(blocked_front)
            if extended:
                lookahead = 0.0
                weight = 1.0
                for index in extended:
                    lookahead += weight * _shifted_distance(
                        wires[index], left, right, logical_to_physical, distances
                    )
                    weight *= LOOKAHEAD_DECAY
                score += LOOKAHEAD_WEIGHT * lookahead / len(extended)
            if best is None or score < best_score:
                best = (left, right)
                best_score = score
        return best

    stalled_limit = STALLED_SWAP_LIMIT_PER_WIRE * n_wires
    while front:
        executable = [
            index for index in front if not blocks[index] or separation(index) == 1
        ]
        if executable:
            for index in executable:
                front.remove(index)
                placements[index] = tuple(
                    logical_to_physical[wire] for wire in wires[index]
                )
                order.append(index)
                executed += 1
                for successor in successors[index]:
                    pending[successor] -= 1
                    if pending[successor] == 0:
                        front.append(successor)
            stalled = 0
            continue

        blocked_front = [index for index in front if blocks[index]]
        choice = best_swap(blocked_front)
        if choice is None:
            raise ValueError(
                "SABRE swap insertion has no candidate: the coupling map is "
                "disconnected across the front layer"
            )
        left, right = choice
        left_logical = physical_to_logical[left]
        right_logical = physical_to_logical[right]
        physical_to_logical[left], physical_to_logical[right] = (
            right_logical,
            left_logical,
        )
        logical_to_physical[left_logical] = right
        logical_to_physical[right_logical] = left
        swaps.append((left, right))
        swap_placements.append(executed)
        swap_anchors.append(blocked_front[0])
        stalled += 1
        if stalled > stalled_limit:
            raise ValueError(
                "SABRE swap insertion made no progress after "
                f"{stalled} consecutive swaps on {n_wires} wires"
            )

    if executed != count:
        raise RuntimeError(
            f"SABRE planned {executed} of {count} instructions; the front layer "
            "emptied before the program was fully placed"
        )
    return SabrePlan(
        sequence=tuple(order),
        placements=tuple(placements),
        swaps=tuple(swaps),
        swap_placements=tuple(swap_placements),
        swap_anchors=tuple(swap_anchors),
        final_logical_to_physical=tuple(logical_to_physical),
    )


__all__ = ["SabrePlan", "plan_sabre_swaps"]
