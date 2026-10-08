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
resolve to the lexicographically smallest physical pair, so the same program and
coupling map always produce the same plan without a seed.

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
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..core.ir import CircuitIR, Instruction
from .ordering_barrier import is_ordering_barrier

if TYPE_CHECKING:
    from .routing import CouplingMap

# SABRE defaults. The extended layer is bounded so planning stays linear in the
# circuit size, and the lookahead weight keeps the front layer dominant.
LOOKAHEAD_DECAY = 0.9
LOOKAHEAD_EXTENDED_SIZE = 20
LOOKAHEAD_WEIGHT = 0.5

# A SWAP that executes no front-layer operation is only useful if it shortens
# some blocked front-layer distance. On a connected topology of N qubits a useful
# SWAP always exists within the graph diameter, so this many consecutive
# stalled SWAPs means the front layer cannot be unblocked.
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
            in a malformed or ambiguous way, or if the front layer cannot be
            unblocked within the stalled-SWAP limit.
    """

    n_wires = program.n_wires
    if coupling.n_qubits < n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")
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

    # One edge per (instruction, qubit) pair, so an instruction whose two qubits
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

    # Classical order is order that no qubit records. Per bit, a read follows the
    # write it observes, and a write follows both the read it would invalidate and
    # the earlier write whose value it replaces. A bit no measurement ever writes
    # contributes no edge, so a program without classical state keeps the
    # qubit-only graph it had. Processing in program order keeps every edge
    # forward, so the graph stays acyclic and the front layer still empties on
    # every program.
    last_writer: dict[int, int] = {}
    last_reader: dict[int, int] = {}
    for index, instruction in enumerate(instructions):
        written, read = _classical_accesses(instruction)
        for bit in read:
            producer = last_writer.get(bit)
            if producer is not None:
                successors[producer].append(index)
                pending[index] += 1
        if written is not None:
            predecessors = (last_writer.get(written), last_reader.pop(written, None))
            for predecessor in predecessors:
                if predecessor is not None:
                    successors[predecessor].append(index)
                    pending[index] += 1
            last_writer[written] = index
        for bit in read:
            last_reader[bit] = index

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
        while logical_to_physical[leaf] != leaf:
            physical = logical_to_physical[leaf]
            step = _restore_path(physical, leaf, tree, active)[1]
            other = physical_to_logical[step]
            logical_to_physical[leaf] = step
            logical_to_physical[other] = physical
            physical_to_logical[step] = leaf
            physical_to_logical[physical] = other
            swaps.append((physical, step) if physical < step else (step, physical))
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
        raise ValueError("Coupling map has fewer wires than the circuit.")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    identity = tuple(range(n_wires))
    if not program.instructions:
        return identity

    reversed_program = CircuitIR(n_wires, tuple(reversed(program.instructions)))
    candidate = identity
    best_layout = identity
    forward = plan_sabre_swaps(program, coupling, initial_layout=candidate)
    best_cost = _restored_swap_count(forward, coupling)
    for _ in range(rounds):
        backward = plan_sabre_swaps(
            reversed_program,
            coupling,
            initial_layout=forward.final_logical_to_physical,
        )
        trial = backward.final_logical_to_physical
        if trial == candidate:
            break
        trial_forward = plan_sabre_swaps(program, coupling, initial_layout=trial)
        trial_cost = _restored_swap_count(trial_forward, coupling)
        if trial_cost >= best_cost:
            break
        best_layout = trial
        best_cost = trial_cost
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
