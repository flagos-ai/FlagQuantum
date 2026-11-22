# Portions adapted from Qiskit 2.6.2, crates/transpiler/src/passes/sabre/route.rs.
# (C) Copyright IBM 2024
# Portions adapted from Cirq 1.7.0, transformers/routing/route_circuit_cqc.py.
# Copyright 2022 The Cirq Developers
# Licensed under the Apache License, Version 2.0. See the repository LICENSE or
# https://www.apache.org/licenses/LICENSE-2.0.
# Modified for FlagQuantum: incidence-indexed native IR scoring, deterministic
# ties, repeated-layout detection, bounded rollback and shortest-path progress.
# Sources: docs/development/COMPILER_UPSTREAM_REUSE.md and
# docs/development/COMPILER_SABRE_RELATIVE_SCORING.md.


"""SABRE-style lookahead SWAP planning for one undirected coupling topology.

The planner implements the SABRE heuristic (Li, Ding, and Xie, "Tackling the
Qubit Mapping Problem for NISQ-Era Quantum Devices", ASPLOS 2019). It keeps a
front layer of two-qubit operations whose predecessors have already executed, and
when no front-layer operation is a coupling edge it inserts the SWAP that
minimizes the mean front-layer hop distance plus a distance-decayed lookahead
term over a bounded extended layer.

The plan is one persistent layout for the whole circuit, so a materializer
replays the forward SWAPs in reverse at the end to restore the logical output
layout. The planner is deterministic: ties between equal-cost SWAP candidates
first use the lexicographically smallest physical pair. Three additional fixed-seed
trials explore equal-cost choices and one layer-lookahead trial considers disjoint
SWAP pairs, retaining a trial only when its complete route,
including final restoration, uses fewer SWAPs. The same input always produces the
same result without a user seed.

Unlike the shortest-path strategies, this planner moves the layout before an
operation rather than immediately before it, which is what makes it effective on
devices of 20 or more qubits.

The planner uses only the qubits the program owns, so it plans on the coupling
subgraph those qubits induce rather than on the whole device. A padding qubit of a
wider device is not an ancilla this planner may swap on, and a program the
circuit's own qubits cannot connect is refused rather than costed against a route
the plan cannot express.

The planner can start from a caller-supplied initial layout, and
``plan_sabre_layout`` searches for one using the layout pass of the same paper:
route the program, route the reversed program from the resulting layout, and
adopt that layout as the next candidate whenever it lowers the total number of
inserted SWAPs. A layout that starts away from the identity cannot be undone by
replaying the forward SWAPs in reverse, so ``restore_swaps`` plans the explicit
restore that returns every logical qubit to its own qubit.

A dependency is not only a shared qubit. A measurement writes classical state that
no qubit records and a conditional reads it, so a conditional and the measurement
that produces the condition it tests are ordered by the classical channel alone,
even when the two touch disjoint qubits. The planner therefore keeps program order
over each classical bit: a read follows the write it observes, and a write follows
the reads it would otherwise invalidate. Without those edges the front layer can
emit a conditional before its measurement, and the dynamic runtime then refuses
the routed program with "classical bit N was read before measurement" — a routing
defect, not a caller error. Classical order is the only channel through which a
reordering of measurements is observable, so a measurement whose bit is neither
re-read nor re-written stays free to move past an operation on other qubits.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...core.ir import CircuitIR, Instruction
from ..optimization.ordering_barrier import is_ordering_barrier
from ._layer_lookahead import choose_layer_swaps

if TYPE_CHECKING:
    from .routing import CouplingMap

# SABRE defaults. The extended layer is bounded so planning stays linear in the
# circuit size, and the lookahead weight keeps the front layer dominant.
LOOKAHEAD_DECAY = 0.9
LOOKAHEAD_EXTENDED_SIZE = 20
LOOKAHEAD_WEIGHT = 0.5

# Bound speculative work between executed instructions. A heuristic can cycle on
# a connected graph; reaching this budget triggers guaranteed shortest-path
# progress rather than misclassifying the circuit as unroutable.
STALLED_SWAP_LIMIT_PER_WIRE = 4

# The layout search reroutes the reversed program until the swap count stops
# improving, so this bound only supplies a hard worst case above the early exits.
SABRE_LAYOUT_ROUNDS = 5


@dataclass(frozen=True)
class SabrePlan:
    """A planned persistent layout and the forward SWAPs that realize it.

    Attributes:
        sequence: Source instruction indices in the order the planner places
            them. A front-layer operation is emitted as soon as its
            predecessors are placed, so this differs from source order when two
            independent operations become executable at different times.
        placements: Physical qubits for every source instruction, indexed by
            source instruction index.
        swaps: Forward SWAP pairs in the order the planner applied them.
        swap_placements: Number of instructions already placed when each forward
            SWAP was applied, so materialization can interleave SWAPs with
            instructions.
        swap_anchors: Source instruction index each forward SWAP is attributed
            to, namely the lowest-indexed blocked front-layer operation when the
            SWAP was chosen. The heuristic picks a SWAP for its effect on the
            whole front layer, so this is an audit anchor, not a sole cause.
        initial_logical_to_physical: Layout the forward SWAPs started from, so a
            caller can tell how far the plan moved away from the identity layout.
        final_logical_to_physical: Layout after the forward SWAPs.
    """

    sequence: tuple[int, ...]
    placements: tuple[tuple[int, ...], ...]
    swaps: tuple[tuple[int, int], ...]
    swap_placements: tuple[int, ...]
    swap_anchors: tuple[int, ...]
    initial_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]


#: The opcode that writes classical state, the metadata key naming the bit it
#: writes, and the two spellings a classical condition reads bits with. No other
#: instruction in the IR writes a classical bit, so a program without a
#: measurement has no classical dependency to preserve.
_MEASURE_OPCODE = "measure"
_CLASSICAL_BIT_KEY = "classical_bit"
_CONDITION_KEYS = ("conditions", "condition_clauses")


def _classical_accesses(
    instruction: Instruction,
) -> tuple[int | None, tuple[int, ...]]:
    """Return the classical bit an instruction writes and the bits it reads.

    The condition spellings are read here exactly as ``runtime.dynamic`` reads
    them. The planner needs the bits itself rather than from that package because
    Compiler may not depend on Runtime, and a metadata read is a smaller cost than
    a cross-layer import.

    Raises:
        ValueError: If the instruction mixes the two condition spellings, or if
            the spelling it uses is not a sequence of bit/value pairs.
    """

    written: int | None = None
    if instruction.name == _MEASURE_OPCODE:
        raw_bit = instruction.metadata.get(_CLASSICAL_BIT_KEY)
        # A measurement that names no bit records nothing a later instruction can
        # read, so it takes no place in the classical order.
        written = None if raw_bit is None else int(raw_bit)

    if all(key in instruction.metadata for key in _CONDITION_KEYS):
        raise ValueError(
            f"instruction {instruction.name}{instruction.wires} cannot define both "
            "conditions and condition_clauses"
        )
    raw_clauses: Any = instruction.metadata.get("condition_clauses")
    if raw_clauses is None and "conditions" in instruction.metadata:
        raw_clauses = (instruction.metadata["conditions"],)
    read: set[int] = set()
    try:
        for clause in raw_clauses or ():
            for bit, _value in clause:
                read.add(int(bit))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"instruction {instruction.name}{instruction.wires} has malformed "
            f"classical conditions: {error}"
        ) from error
    return written, tuple(sorted(read))


def _validated_layout(
    layout: tuple[int, ...] | None,
    n_wires: int,
) -> tuple[int, ...]:
    """Return a caller layout as a placement, defaulting to the identity."""

    if layout is None:
        return tuple(range(n_wires))
    placement = tuple(layout)
    if len(placement) != n_wires or sorted(placement) != list(range(n_wires)):
        raise ValueError(
            "initial_layout must be a permutation of the circuit wires, got "
            f"{placement!r} for {n_wires} wires"
        )
    return placement


def _planning_coupling(coupling: CouplingMap, n_wires: int) -> CouplingMap:
    """Return the coupling subgraph induced by the qubits the program owns.

    The planner emits SWAPs only on those qubits, so the graph it plans on has to be
    the same graph. A padding qubit of a wider device is not an ancilla this planner
    can swap on, and scoring candidates against device distances that only such a
    qubit realizes would cost the plan against a route it cannot emit.
    """

    if coupling.n_qubits == n_wires:
        return coupling
    # ``routing`` imports this module, so the value type is imported here.
    from .routing import CouplingMap as _CouplingMap

    return _CouplingMap(
        n_wires,
        tuple(edge for edge in coupling.edges if edge[1] < n_wires),
    )


def _shifted_distance(
    wire_pair: tuple[int, ...],
    left: int,
    right: int,
    logical_to_physical: list[int],
    distances: tuple[tuple[float, ...], ...],
) -> float:
    """Return a two-qubit hop distance as it would be after a candidate SWAP."""

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


def _relative_score_swap(
    candidates: set[tuple[int, int]],
    qubit_pairs: tuple[tuple[int, ...], ...],
    front: list[int],
    extended: list[int],
    layout: list[int],
    distances: tuple[tuple[float, ...], ...],
    *,
    tie_rng: random.Random | None = None,
) -> tuple[int, int] | None:
    """Score only interactions incident on either endpoint of a proposed SWAP."""

    def absolute_score(edge: tuple[int, int]) -> float:
        left, right = edge

        def shifted(index: int) -> float:
            return _shifted_distance(qubit_pairs[index], left, right, layout, distances)

        score = sum(shifted(index) for index in front) / len(front)
        if extended:
            lookahead = 0.0
            weight = 1.0
            for index in extended:
                lookahead += weight * shifted(index)
                weight *= LOOKAHEAD_DECAY
            score += LOOKAHEAD_WEIGHT * lookahead / len(extended)
        return score

    ordered = sorted(candidates)
    if tie_rng is not None:
        tie_rng.shuffle(ordered)
    if not ordered:
        return None
    terms: list[tuple[int, int, float]] = []
    incidence: list[list[tuple[int, float]]] = [[] for _ in layout]
    for indices, scale in (
        (front, 1.0 / len(front)),
        (extended, LOOKAHEAD_WEIGHT / len(extended) if extended else 0.0),
    ):
        weight = 1.0
        for index in indices:
            first, second = (layout[wire] for wire in qubit_pairs[index])
            coefficient = scale * weight
            terms.append((first, second, coefficient))
            if first != second:
                incidence[first].append((second, coefficient))
                incidence[second].append((first, coefficient))
            if indices is extended:
                weight *= LOOKAHEAD_DECAY
    # All candidates share the omitted current-layout score. Near a rounding
    # tie, replay the original absolute arithmetic to preserve its tie behavior.
    magnitude = math.fsum(weight * distances[a][b] for a, b, weight in terms)
    tie_tolerance = 4 * len(terms) * math.ulp(max(1.0, magnitude))
    best: tuple[int, int] | None = None
    best_score = 0.0
    for left, right in ordered:
        changes = []
        # Coupling distances are symmetric. An interaction touching both
        # endpoints keeps its distance; every other incident term changes only
        # its moved endpoint. Preserve one contribution per source term.
        for moved, destination in ((left, right), (right, left)):
            before, after = distances[moved], distances[destination]
            for other, coefficient in incidence[moved]:
                if other != destination:
                    changes.append(coefficient * (after[other] - before[other]))
        score = math.fsum(changes)
        if best is None:
            best, best_score = (left, right), score
        elif abs(score - best_score) <= tie_tolerance:
            if absolute_score((left, right)) < absolute_score(best):
                best, best_score = (left, right), score
        elif score < best_score:
            best, best_score = (left, right), score
    return best


def plan_sabre_swaps(
    program: CircuitIR,
    coupling: CouplingMap,
    *,
    initial_layout: tuple[int, ...] | None = None,
) -> SabrePlan:
    """Plan a SABRE persistent layout for one program on one coupling map.

    Only the qubits owned by the program are used, and inserted SWAPs always join
    two qubits of the program, so no physical ancilla is required. Planning happens
    on the coupling subgraph those qubits induce, so a wider device is supported
    exactly as far as its qubits inside the circuit connect the program.

    Args:
        program: Circuit to place.
        coupling: Topology the placed circuit must respect.
        initial_layout: Physical qubit for each logical qubit at the start. Defaults
            to the identity layout.

    Raises:
        ValueError: If the coupling map has fewer qubits than the program, if
            ``initial_layout`` is not a permutation of the circuit qubits, if a
            two-qubit operation is connected on the device only through qubits the
            circuit does not own, if an instruction states its classical condition
            in a malformed or ambiguous way.
    """

    baseline = _plan_sabre_trial(program, coupling, initial_layout=initial_layout)
    if baseline is None:
        raise RuntimeError("unbounded SABRE trial did not produce a plan")
    return _select_sabre_trials(program, coupling, baseline)


def _select_sabre_trials(
    program: CircuitIR,
    coupling: CouplingMap,
    baseline: SabrePlan,
    *,
    cost: Callable[[SabrePlan], int] | None = None,
) -> SabrePlan:
    """Retain the single-trial route unless a bounded trial is strictly cheaper."""
    if not baseline.swaps:
        return baseline
    return min(
        _sabre_trial_costs(program, coupling, baseline, cost=cost),
        key=lambda item: item[1],
    )[0]


def _sabre_trial_costs(
    program: CircuitIR,
    coupling: CouplingMap,
    baseline: SabrePlan,
    *,
    cost: Callable[[SabrePlan], int] | None = None,
) -> Iterator[tuple[SabrePlan, int]]:
    """Yield completed trials once, preserving incumbent-based pruning and ties.

    Consumers may inspect alternative plans without repeating search. A pruned
    trial is absent, so this is not an enumeration of all equal-cost routes.
    Dependencies and plans remain owned by this invocation.
    """
    score = (
        cost if cost is not None else lambda plan: _restored_swap_count(plan, coupling)
    )
    best_cost = score(baseline)
    yield baseline, best_cost
    if not baseline.swaps:
        return
    dependencies = _build_sabre_dependencies(program.instructions)
    for seed, use_layers in ((0, False), (1, False), (2, False), (None, True)):
        candidate = _plan_sabre_trial(
            program,
            coupling,
            initial_layout=baseline.initial_logical_to_physical,
            tie_rng=random.Random(seed) if seed is not None else None,
            committed_swap_limit=best_cost,
            layer_lookahead=use_layers,
            _dependencies=dependencies,
        )
        if candidate is not None:
            candidate_cost = score(candidate)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
            yield candidate, candidate_cost


def _build_sabre_dependencies(
    instructions: tuple[Instruction, ...],
) -> tuple[list[list[int]], tuple[int, ...]]:
    """Build read-only successors and initial indegrees for one source program."""
    # One edge per (instruction, qubit) pair, so an instruction whose two qubits
    # share a predecessor still reaches zero pending predecessors.
    successors: list[list[int]] = [[] for _ in range(len(instructions))]
    pending = [0] * len(instructions)
    last_on_wire: dict[int, int] = {}
    for index, instruction in enumerate(instructions):
        wire_pair = instruction.wires
        for wire in wire_pair:
            previous = last_on_wire.get(wire)
            if previous is not None:
                successors[previous].append(index)
                pending[index] += 1
            last_on_wire[wire] = index

    # Classical order is order that no qubit records. Per bit, a read follows the
    # write it observes, and a write follows every read it would invalidate and
    # the earlier write whose value it replaces. A bit no measurement ever writes
    # contributes no edge, so a program without classical state keeps the
    # qubit-only graph it had. Processing in program order keeps every edge
    # forward, so the graph stays acyclic and the front layer still empties on
    # every program.
    last_writer: dict[int, int] = {}
    # Readers on independent qubits have no quantum edge ordering them; keeping
    # only the most recent reader would let a write overtake an earlier reader.
    readers_since_write: dict[int, list[int]] = {}
    for index, instruction in enumerate(instructions):
        written, read = _classical_accesses(instruction)
        for bit in read:
            producer = last_writer.get(bit)
            if producer is not None:
                successors[producer].append(index)
                pending[index] += 1
        if written is not None:
            predecessors = (
                last_writer.get(written),
                *readers_since_write.pop(written, ()),
            )
            for predecessor in predecessors:
                if predecessor is not None:
                    successors[predecessor].append(index)
                    pending[index] += 1
            last_writer[written] = index
        for bit in read:
            readers_since_write.setdefault(bit, []).append(index)

    return successors, tuple(pending)


def _plan_sabre_trial(
    program: CircuitIR,
    coupling: CouplingMap,
    *,
    initial_layout: tuple[int, ...] | None = None,
    tie_rng: random.Random | None = None,
    committed_swap_limit: int | None = None,
    layer_lookahead: bool = False,
    _dependencies: tuple[list[list[int]], tuple[int, ...]] | None = None,
) -> SabrePlan | None:
    """Run one search; prune only committed SWAPs that cannot beat the incumbent."""

    n_wires = program.n_wires
    if coupling.n_qubits < n_wires:
        raise ValueError("Coupling map has fewer qubits than the circuit.")
    placement = _validated_layout(initial_layout, n_wires)
    instructions = program.instructions
    count = len(instructions)
    if count == 0:
        return SabrePlan(
            sequence=(),
            placements=(),
            swaps=(),
            swap_placements=(),
            swap_anchors=(),
            initial_logical_to_physical=placement,
            final_logical_to_physical=placement,
        )

    # ``CouplingMap.distance_matrix`` marks a disconnected pair with a negative
    # sentinel. A cost function needs such a pair to be maximally expensive
    # rather than maximally cheap, so a local infinite-distance copy is used.
    planning = _planning_coupling(coupling, n_wires)
    distances = tuple(
        tuple(math.inf if hop < 0 else float(hop) for hop in row)
        for row in planning.distance_matrix()
    )
    neighbours = tuple(planning.neighbors(wire) for wire in range(n_wires))
    wires = tuple(instruction.wires for instruction in instructions)
    # A channel occupies its qubits but must not be pulled onto a coupling edge,
    # so it never blocks the front layer. A barrier is in the same position for
    # the same reason: neither one is a unitary the device has to host, so making
    # either adjacent would buy nothing.
    blocks = tuple(
        len(wire_pair) == 2
        and not instruction.metadata.get("is_channel")
        and not is_ordering_barrier(instruction)
        for instruction, wire_pair in zip(instructions, wires, strict=True)
    )
    # A SWAP along one coupling edge keeps each logical qubit inside its own
    # connected component, so a two-qubit operation whose logical qubits have no
    # coupling path between them can never be legalized. Connectivity is a
    # property of the physical qubits, so the initial layout is applied first.
    for index, wire_pair in enumerate(wires):
        if not blocks[index]:
            continue
        left = placement[wire_pair[0]]
        right = placement[wire_pair[1]]
        if not math.isinf(distances[left][right]):
            continue
        if planning is not coupling and coupling.distance(left, right) >= 0:
            raise ValueError(
                f"the device connects wires {wire_pair[0]} and {wire_pair[1]} of "
                f"two-wire instruction {index} only through physical ancilla wires "
                "outside the circuit IR, which this planner does not route through; "
                "provide an explicit layout/lowering step"
            )
        raise ValueError(
            f"no coupling path between wires {wire_pair[0]} and "
            f"{wire_pair[1]} of two-wire instruction {index}"
        )

    successors, initial_pending = (
        _build_sabre_dependencies(instructions)
        if _dependencies is None
        else _dependencies
    )
    # Successors are read-only; each trial consumes its own remaining indegrees.
    pending = list(initial_pending)

    logical_to_physical = list(placement)
    physical_to_logical = [0] * n_wires
    for logical, physical in enumerate(logical_to_physical):
        physical_to_logical[physical] = logical
    placements: list[tuple[int, ...]] = [() for _ in range(count)]
    order: list[int] = []
    swaps: list[tuple[int, int]] = []
    swap_placements: list[int] = []
    swap_anchors: list[int] = []
    front = [index for index in range(count) if pending[index] == 0]
    executed = 0
    stalled = 0
    progress_swap_start = 0
    seen_layouts: set[tuple[int, ...]] = set()
    frontier_lookahead: list[int] | None = None

    def exchange(left: int, right: int) -> None:
        left_logical = physical_to_logical[left]
        right_logical = physical_to_logical[right]
        physical_to_logical[left], physical_to_logical[right] = (
            right_logical,
            left_logical,
        )
        logical_to_physical[left_logical] = right
        logical_to_physical[right_logical] = left

    def record_swap(left: int, right: int, anchor: int) -> None:
        exchange(left, right)
        swaps.append((left, right))
        swap_placements.append(executed)
        swap_anchors.append(anchor)

    def separation(index: int) -> float:
        left = logical_to_physical[wires[index][0]]
        right = logical_to_physical[wires[index][1]]
        return distances[left][right]

    def extended_layer(blocked_front: list[int]) -> list[int]:
        """Return blocked operations reachable from the front, in BFS order."""
        nonlocal frontier_lookahead
        # SWAPs change placement, not dependency reachability. Rebuild only when
        # a source instruction executes and changes the frontier.
        if frontier_lookahead is not None:
            return frontier_lookahead
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
        frontier_lookahead = extended
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
        if n_wires > 12:
            return _relative_score_swap(
                candidates,
                wires,
                blocked_front,
                extended_layer(blocked_front),
                logical_to_physical,
                distances,
                tie_rng=tie_rng,
            )
        # Preserve the lower-overhead full scoring path on small devices.
        extended = extended_layer(blocked_front)
        best: tuple[int, int] | None = None
        best_score = 0.0
        ordered = sorted(candidates)
        if tie_rng is not None:
            tie_rng.shuffle(ordered)
        for left, right in ordered:
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
        # A speculative suffix may still be rolled back by progress recovery.
        # Only the prefix committed by an executed instruction is a lower bound
        # on the final cost; using len(swaps) here would incorrectly prune it.
        if (
            committed_swap_limit is not None
            and progress_swap_start >= committed_swap_limit
        ):
            return None
        executable = [
            index for index in front if not blocks[index] or separation(index) == 1
        ]
        if executable:
            frontier_lookahead = None
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
            progress_swap_start = len(swaps)
            seen_layouts.clear()
            continue

        blocked_front = [index for index in front if blocks[index]]
        layout = tuple(logical_to_physical)
        if layout in seen_layouts or stalled >= stalled_limit:
            # Qiskit's release valve undoes speculation since the last routed
            # instruction. No source instruction has executed in this suffix,
            # so removing it preserves both semantics and attribution.
            for left, right in reversed(swaps[progress_swap_start:]):
                exchange(left, right)
            del swaps[progress_swap_start:]
            del swap_placements[progress_swap_start:]
            del swap_anchors[progress_swap_start:]
            anchor = min(blocked_front, key=lambda index: (separation(index), index))
            first, second = wires[anchor]
            path = planning.shortest_path(
                logical_to_physical[first], logical_to_physical[second]
            )
            # Bring both ends together, stopping one edge short. Every SWAP
            # reduces their distance by one; the next iteration must execute
            # this front-layer instruction. Unrelated dependencies stay intact.
            required = len(path) - 2
            for offset in range((required + 1) // 2):
                record_swap(path[offset], path[offset + 1], anchor)
            for offset in range(required // 2):
                end = len(path) - 1 - offset
                record_swap(path[end], path[end - 1], anchor)
            continue
        # Cirq tracks repeated search choices before forcing a shortest path.
        # Track complete layouts here because FQ's frontier is a dependency DAG,
        # rather than Cirq's disjoint two-qubit timesteps.
        seen_layouts.add(layout)
        if layer_lookahead:
            candidates = {
                (min(physical, neighbor), max(physical, neighbor))
                for index in blocked_front
                for qubit in wires[index]
                for physical in (logical_to_physical[qubit],)
                for neighbor in neighbours[physical]
            }
            moves = choose_layer_swaps(
                candidates,
                front,
                wires,
                blocks,
                successors,
                pending,
                logical_to_physical,
                distances,
            )
            if moves is not None:
                for left, right in moves:
                    record_swap(left, right, blocked_front[0])
                stalled += len(moves)
                continue
        choice = best_swap(blocked_front)
        if choice is None:
            raise ValueError(
                "SABRE swap insertion has no candidate: the coupling map is "
                "disconnected across the front layer"
            )
        left, right = choice
        record_swap(left, right, blocked_front[0])
        stalled += 1

    if executed != count:
        raise RuntimeError(
            f"SABRE planned {executed} of {count} instructions; the front layer "
            "emptied before the program was fully placed"
        )
    # A materializer replays the SWAPs in order while it walks ``sequence``, so
    # the plan is only usable if that replay reproduces the layout the planner
    # costed. Fail closed here rather than let a caller route against a layout
    # that never existed.
    replay_logical_to_physical = list(placement)
    replay_physical_to_logical = [0] * n_wires
    for logical, physical in enumerate(replay_logical_to_physical):
        replay_physical_to_logical[physical] = logical
    for swap_left, swap_right in swaps:
        left_logical = replay_physical_to_logical[swap_left]
        right_logical = replay_physical_to_logical[swap_right]
        replay_physical_to_logical[swap_left] = right_logical
        replay_physical_to_logical[swap_right] = left_logical
        replay_logical_to_physical[left_logical] = swap_right
        replay_logical_to_physical[right_logical] = swap_left
    if replay_logical_to_physical != logical_to_physical:
        raise RuntimeError(
            "SABRE plan does not replay to its planned layout; the reported "
            "layout and the reported SWAPs disagree"
        )
    return SabrePlan(
        sequence=tuple(order),
        placements=tuple(placements),
        swaps=tuple(swaps),
        swap_placements=tuple(swap_placements),
        swap_anchors=tuple(swap_anchors),
        initial_logical_to_physical=placement,
        final_logical_to_physical=tuple(logical_to_physical),
    )


def _restore_path(
    start: int,
    goal: int,
    tree: list[list[int]],
    active: list[bool],
) -> list[int]:
    """Return an active graph path from ``start`` to ``goal``, inclusive."""

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
    restore. One leaf of a spanning forest at a time is walked home without
    displacing fixed qubits. On a graph with cycles, a second walk can use all
    remaining coupling edges. It replaces the tree walk only if it is shorter.

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

    baseline = _walk_restore(placement, tree, tree)
    if sum(map(len, adjacency)) == sum(map(len, tree)):
        return baseline
    shortcuts = _walk_restore(placement, tree, adjacency)
    return shortcuts if len(shortcuts) < len(baseline) else baseline


def _walk_restore(
    placement: tuple[int, ...],
    tree: list[list[int]],
    paths: list[list[int]],
) -> tuple[tuple[int, int], ...]:
    """Fix tree leaves using paths in a graph containing that tree.

    Removing a tree leaf preserves connectivity of the remaining component.
    Paths exclude fixed vertices, so shortcuts cannot displace restored tokens.
    Both walks terminate after exactly one leaf removal per circuit wire.
    """

    n_wires = len(placement)
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
        # Only token positions change while walking; the active graph is fixed.
        path = _restore_path(physical, leaf, paths, active) if physical != leaf else ()
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


def _restored_swap_count(plan: SabrePlan, coupling: CouplingMap) -> int:
    """Return the SWAPs a forward plan inserts together with its restore SWAPs."""

    return len(plan.swaps) + len(plan_restore_swaps(plan, coupling))


def plan_sabre_layout(
    program: CircuitIR,
    coupling: CouplingMap,
    *,
    rounds: int = SABRE_LAYOUT_ROUNDS,
) -> tuple[int, ...]:
    """Search for an initial layout that shortens the SABRE-routed program.

    The search follows the layout pass of the SABRE paper: route the program,
    route the reversed program starting from the layout that left behind, and
    adopt that layout as the next candidate. A candidate replaces the current best
    only when it lowers the inserted-SWAP count of a fully restored route, so the
    returned layout never costs more than the identity layout.

    Candidates are searched on the coupling subgraph the program's own qubits
    induce, so a device wider than the program is searched exactly as far as those
    qubits connect it.

    Raises:
        ValueError: If the coupling map has fewer qubits than the program, if
            ``rounds`` is negative, or if the program cannot be routed.
    """

    n_wires = program.n_wires
    if coupling.n_qubits < n_wires:
        raise ValueError("Coupling map has fewer qubits than the circuit.")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    identity = tuple(range(n_wires))
    if not program.instructions:
        return identity

    reversed_program = CircuitIR(n_wires, tuple(reversed(program.instructions)))
    candidate = identity
    best_layout = identity
    # Keep the original deterministic forward/backward trajectory as candidate
    # generation. Substituting a better forward route here can lead the reverse
    # pass away from a valuable old layout. Evaluate each old candidate with the
    # multi-trial planner, while retaining identity as a candidate throughout.
    forward = _plan_sabre_trial(program, coupling, initial_layout=candidate)
    if forward is None:
        raise RuntimeError("unbounded SABRE trial did not produce a plan")
    trajectory_cost = _restored_swap_count(forward, coupling)
    best_cost = _restored_swap_count(
        _select_sabre_trials(program, coupling, forward), coupling
    )
    for _ in range(rounds):
        backward = _plan_sabre_trial(
            reversed_program,
            coupling,
            initial_layout=forward.final_logical_to_physical,
        )
        if backward is None:
            raise RuntimeError("unbounded SABRE trial did not produce a plan")
        trial = backward.final_logical_to_physical
        if trial == candidate:
            break
        trial_forward = _plan_sabre_trial(program, coupling, initial_layout=trial)
        if trial_forward is None:
            raise RuntimeError("unbounded SABRE trial did not produce a plan")
        trial_cost = _restored_swap_count(trial_forward, coupling)
        improved = _select_sabre_trials(program, coupling, trial_forward)
        improved_cost = _restored_swap_count(improved, coupling)
        if improved_cost < best_cost:
            best_layout, best_cost = trial, improved_cost
        if trial_cost >= trajectory_cost:
            break
        trajectory_cost = trial_cost
        candidate = trial
        forward = trial_forward
    return best_layout


__all__ = [
    "SabrePlan",
    "plan_restore_swaps",
    "plan_sabre_layout",
    "plan_sabre_swaps",
    "restore_swaps",
]
