"""Measure the lookahead SWAP planner Qiskit ships against the shipped planners.

W7-04 of the Qiskit parity backlog asks for Qiskit's ``LookaheadSwap``, whose
design is Jandura's 2018 swap mapper: rank every candidate SWAP by the layout it
produces, recurse into the best ``search_width`` of them to ``search_depth``
levels, and keep the chain whose step score is best. The step score is Qiskit's:
the two-wire operations the chain makes executable, less three per inserted SWAP.

This module builds that planner and measures it through the shipped consumer
(``route_to_topology``) and the shipped optimizer, so every comparison is the same
code path and the same cost model on both sides. The answer is negative, and the
measurement is checked in so the decision can be re-tested rather than re-argued.
On the 140-program basis this module measures by default, whose shipped baselines
retain 2410 SWAPs (``sabre``) and 2051 (``sabre_layout``) after optimization:

* The reference search is *correct*. Every compiled program is compared with its
  source state (maximum difference 2.8e-16 over 1217 compiled programs), no
  configuration emits a two-wire instruction off the device, and at search depth
  and width one the search reproduces the shipped planner's inserted-SWAP sequence
  on 139 of 140 programs and its planned SWAP count on 140 of 140. The port is a
  faithful replacement, not an approximation of one.
* Qiskit's own objective -- the summed separation over the front layer plus a
  bounded breadth-first extension of it -- does not terminate on every program. At
  its documented defaults it retains 1122 SWAPs, less than half of ``sabre``, but
  it *fails closed on 24 of the 140 programs*, because it drives the plan back to
  a state it already held while an operation is still blocked. Two other
  objectives (``front``, ``sabre``) plan all 140, including all 24 the published
  objective cannot finish, so the failure belongs to that ranking rather than to
  the search. Repairing it with the single SWAP the shipped planner would take
  makes it total, and 23 % *worse* than ``sabre_layout``.
* No objective beats ``sabre_layout``. The best of them retains 2344 SWAPs: 2.7 %
  better than the greedy ``sabre`` it generalizes, and 14 % worse than the
  ``sabre_layout`` strategy the product already selects.
* Substituting the search into the shipped *layout* search is the closest the port
  comes, and the gain does not replicate: 0.982 of ``sabre_layout`` on the recorded
  basis, 0.997 and 0.976 on two disjoint seed sets, for ten to twenty times the
  planning time. The sign of that difference is not stable.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_lookahead_swap.py --json-output /tmp/lookahead.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

import flagquantum as fq
import flagquantum.compiler.routing as routing_module
import flagquantum.compiler.sabre as sabre_module
from flagquantum.compiler import CouplingMap, optimize, route_to_topology
from flagquantum.compiler.routing import record_post_routing_optimization
from flagquantum.compiler.sabre import (
    LOOKAHEAD_DECAY,
    LOOKAHEAD_EXTENDED_SIZE,
    LOOKAHEAD_WEIGHT,
    STALLED_SWAP_LIMIT_PER_WIRE,
    SabrePlan,
    _planning_coupling,
    _validated_layout,
    plan_restore_swaps,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode

# Name -> (wires, edges, program widths). Every topology is measured at a program
# narrower than the device, where a placement can matter, and at a program as wide
# as the device, where only routing can.
_TOPOLOGIES: dict[str, tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]] = {
    "line4": (4, CouplingMap.line(4).edges, (2, 4)),
    "ring5": (5, CouplingMap.ring(5).edges, (3, 5)),
    "line6": (6, CouplingMap.line(6).edges, (3, 6)),
    "ring6": (6, CouplingMap.ring(6).edges, (3, 6)),
    "grid3x3": (9, CouplingMap.grid(3, 3).edges, (6, 9)),
    "line9": (9, CouplingMap.line(9).edges, (5, 9)),
    "grid4x4": (16, CouplingMap.grid(4, 4).edges, (8, 16)),
}
SEEDS = (0, 1, 2, 3, 4)
GATE_FACTORS = (2, 6)
_ROTATIONS = ("rx", "ry", "rz")
_TWO_WIRE = ("cx", "cz")

OBJECTIVES = ("published", "front", "sabre")

# A disconnected physical pair cannot host a two-wire operation at all.
# ``CouplingMap.distance_matrix`` marks it with a negative sentinel, and a cost
# function needs it to sort last rather than first, so it becomes a finite value
# above any real hop count.
_UNREACHABLE_PENALTY = 1e6


@dataclass(frozen=True)
class Configuration:
    """One planner to measure against the shipped strategies.

    Attributes:
        label: Stable name of the row this configuration produces.
        strategy: Shipped routing strategy whose consumer materializes the plan.
        objective: ``None`` keeps the shipped SWAP search; every other value
            selects a reference objective. ``published`` is Qiskit's
            summed-distance window, ``front`` is the min-max front-layer key, and
            ``sabre`` reads the shipped planner's cost on a whole layout.
        search_depth: SWAPs in a candidate chain.
        search_width: Candidate chains expanded at each level.
        repair: Replace a chain that returns to a held state with the single SWAP
            the shipped planner would take, instead of failing closed.
    """

    label: str
    strategy: str
    objective: str | None = None
    search_depth: int = 0
    search_width: int = 0
    repair: bool = False

    def planner(self) -> Any:
        """Return the SWAP search this configuration routes with."""

        if self.objective is None:
            return None

        def plan(
            program: CircuitIR,
            coupling: CouplingMap,
            *,
            initial_layout: tuple[int, ...] | None = None,
        ) -> SabrePlan:
            return plan_lookahead_swaps(
                program,
                coupling,
                initial_layout=initial_layout,
                search_depth=self.search_depth,
                search_width=self.search_width,
                objective=self.objective,
                repair=self.repair,
            )

        return plan


DEFAULT_CONFIGURATIONS = (
    Configuration("sabre", "sabre"),
    Configuration("sabre_layout", "sabre_layout"),
    # The fidelity anchor: the reference search at its smallest setting, where the
    # shipped planner's own decision rule is the only candidate it can consider.
    Configuration("sabre/d1w1", "sabre", "sabre", 1, 1),
    Configuration("published/d4w4", "sabre", "published", 4, 4),
    Configuration("published/d4w6", "sabre", "published", 4, 6),
    Configuration("published/d4w4+repair", "sabre", "published", 4, 4, repair=True),
    Configuration("published/d4w6+repair", "sabre", "published", 4, 6, repair=True),
    Configuration("front/d4w4", "sabre", "front", 4, 4),
    Configuration("sabre/d3w2", "sabre", "sabre", 3, 2),
    Configuration("layout/published/d4w4", "sabre_layout", "published", 4, 4),
    Configuration("layout/sabre/d2w2", "sabre_layout", "sabre", 2, 2),
    Configuration("layout/sabre/d3w2", "sabre_layout", "sabre", 3, 2),
    Configuration("layout/sabre/d4w2", "sabre_layout", "sabre", 4, 2),
    Configuration("layout/sabre/d4w4", "sabre_layout", "sabre", 4, 4),
)


@dataclass(frozen=True)
class _Step:
    """One chain of SWAPs, the layout it reaches, and what it unblocks."""

    chain: tuple[tuple[int, int], ...]
    layout: tuple[int, ...]
    blocked: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class _Case:
    """One deterministic program on one device."""

    topology: str
    n_wires: int
    edges: tuple[tuple[int, int], ...]
    program_wires: int
    seed: int
    gate_count: int

    @property
    def label(self) -> str:
        return f"{self.topology}/w{self.program_wires}/s{self.seed}/g{self.gate_count}"

    def device(self) -> CouplingMap:
        """Return a device that no earlier case has routed on."""

        return CouplingMap(self.n_wires, self.edges)

    def program(self) -> CircuitIR:
        return seeded_program(self.seed, self.program_wires, self.gate_count)


def seeded_program(seed: int, n_wires: int, gate_count: int) -> CircuitIR:
    """Return a deterministic mixed single- and two-wire program.

    Two-wire operands are drawn from every wire pair, so most programs start with
    operations the device cannot execute directly.
    """

    rng = random.Random(seed)
    instructions: list[Instruction] = []
    for _ in range(gate_count):
        if rng.random() < 0.35:
            instructions.append(
                Instruction(
                    rng.choice(_ROTATIONS),
                    (rng.randrange(n_wires),),
                    params={"theta": rng.uniform(-3.0, 3.0)},
                )
            )
        else:
            left, right = rng.sample(range(n_wires), 2)
            instructions.append(Instruction(rng.choice(_TWO_WIRE), (left, right)))
    return CircuitIR(
        n_wires,
        tuple(instructions),
        dtype="complex128",
        measurements=(MeasurementNode("sample", tuple(range(n_wires)), shots=4),),
    )


def case_basis(
    *,
    topologies: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[_Case, ...]:
    """Return the measured programs, ordered by topology, width, seed, and size."""

    names = tuple(_TOPOLOGIES) if topologies is None else topologies
    unknown = [name for name in names if name not in _TOPOLOGIES]
    if unknown:
        raise ValueError(f"unknown topologies {unknown}; known: {list(_TOPOLOGIES)}")
    return tuple(
        _Case(name, n_wires, edges, program_wires, seed, factor * program_wires)
        for name in names
        for n_wires, edges, widths in (_TOPOLOGIES[name],)
        for program_wires in widths
        for seed in seeds
        for factor in GATE_FACTORS
    )


def planned_swap_count(plan: SabrePlan, coupling: CouplingMap) -> int:
    """Return the forward and restore SWAPs a plan inserts."""

    return len(plan.swaps) + len(plan_restore_swaps(plan, coupling))


def plan_lookahead_swaps(
    program: CircuitIR,
    coupling: CouplingMap,
    *,
    initial_layout: tuple[int, ...] | None = None,
    search_depth: int = 4,
    search_width: int = 4,
    objective: str = "published",
    repair: bool = False,
) -> SabrePlan:
    """Plan a persistent layout by searching chains of SWAPs.

    The plan has the shape :class:`~flagquantum.compiler.sabre.SabrePlan` gives
    the shipped planner, so the shipped materialization, restore, and cost model
    all apply to it unchanged.

    Args:
        program: The program to place and route.
        coupling: The device to route on.
        initial_layout: Logical-to-physical placement to start from.
        search_depth: SWAPs in a candidate chain.
        search_width: Candidate chains expanded at each level.
        objective: ``published``, ``front``, or ``sabre``; see the module
            docstring for what each one measures.
        repair: Replace a chain that returns to a state the planner already held
            with the single SWAP the shipped planner would take, rather than
            raising.

    Raises:
        ValueError: If an argument is out of range, if the program has no coupling
            path for a two-wire operation, if no chain of distinct coupling edges
            unblocks the front layer, or if a chain returns to a held state while
            ``repair`` is false.
    """

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    if search_depth < 1:
        raise ValueError("search_depth must be at least one")
    if search_width < 1:
        raise ValueError("search_width must be at least one")

    n_wires = program.n_wires
    if coupling.n_qubits < n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")
    placement = _validated_layout(initial_layout, n_wires)
    instructions = program.instructions
    count = len(instructions)
    if count == 0:
        return SabrePlan((), (), (), (), (), placement, placement)

    # The plan may only swap between wires the program owns, so it is costed on
    # the coupling subgraph those wires induce -- the same graph, and the same
    # helpers, the shipped planner uses.
    planning = _planning_coupling(coupling, n_wires)
    distances = tuple(
        tuple(_UNREACHABLE_PENALTY if hop < 0 else float(hop) for hop in row)
        for row in planning.distance_matrix()
    )
    edges = tuple(sorted(planning.edges))
    wires = tuple(instruction.wires for instruction in instructions)
    blocks = tuple(
        len(wire_pair) == 2 and not instruction.metadata.get("is_channel")
        for instruction, wire_pair in zip(instructions, wires, strict=True)
    )
    for index, wire_pair in enumerate(wires):
        if not blocks[index]:
            continue
        if distances[placement[wire_pair[0]]][placement[wire_pair[1]]] < (
            _UNREACHABLE_PENALTY
        ):
            continue
        raise ValueError(
            f"no coupling path between wires {wire_pair[0]} and "
            f"{wire_pair[1]} of two-wire instruction {index}"
        )

    # Per-wire dependency order, not source order: an operation is executable once
    # everything before it on its own wires is placed.
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

    def swapped(layout: tuple[int, ...], left: int, right: int) -> tuple[int, ...]:
        trial = list(layout)
        for logical in range(n_wires):
            if trial[logical] == left:
                trial[logical] = right
            elif trial[logical] == right:
                trial[logical] = left
        return tuple(trial)

    def separations(layout: tuple[int, ...], gates: tuple[int, ...]) -> list[float]:
        return [
            distances[layout[wires[index][0]]][layout[wires[index][1]]]
            for index in gates
        ]

    def extension(blocked: tuple[int, ...]) -> tuple[int, ...]:
        """Return blocked operations reachable from the front, breadth first."""

        extended: list[int] = []
        seen = set(blocked)
        queue: list[int] = []
        for index in blocked:
            for successor in successors[index]:
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        cursor = 0
        while cursor < len(queue) and len(extended) < LOOKAHEAD_EXTENDED_SIZE:
            index = queue[cursor]
            cursor += 1
            if blocks[index]:
                extended.append(index)
            for successor in successors[index]:
                if successor not in seen:
                    seen.add(successor)
                    queue.append(successor)
        return tuple(extended)

    def key(
        layout: tuple[int, ...], blocked: tuple[int, ...], goal: str
    ) -> tuple[float, ...]:
        """Return the objective value for one layout; smaller is better."""

        front = separations(layout, blocked)
        beyond = extension(blocked)
        wide = separations(layout, beyond)
        if goal == "front":
            # Driving the largest separation to one hop is what makes a blocked
            # operation executable, so it is ranked before the sums.
            return (max(front), sum(front), sum(wide))
        if goal == "published":
            # Qiskit's "layout distance" over the front layer and its extension.
            return (sum(front) + sum(wide),)
        # The shipped planner's own cost, read on a whole layout instead of on one
        # candidate SWAP, so the only difference from it is greedy versus search.
        mean = sum(front) / len(front)
        if not wide:
            return (mean,)
        weighted = 0.0
        weight = 1.0
        for value in wide:
            weighted += weight * value
            weight *= LOOKAHEAD_DECAY
        return (mean + LOOKAHEAD_WEIGHT * weighted / len(wide),)

    def advance(
        layout: tuple[int, ...],
        front: tuple[int, ...],
        waiting: list[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...], list[int]]:
        """Place every executable front-layer operation, in front-layer order."""

        order: list[int] = []
        current = list(front)
        waiting = list(waiting)
        changed = True
        while changed:
            changed = False
            for index in list(current):
                if blocks[index] and separations(layout, (index,))[0] != 1.0:
                    continue
                current.remove(index)
                order.append(index)
                for successor in successors[index]:
                    waiting[successor] -= 1
                    if waiting[successor] == 0:
                        current.append(successor)
                changed = True
        return tuple(order), tuple(current), waiting

    def search(
        layout: tuple[int, ...],
        blocked: tuple[int, ...],
        waiting: list[int],
        depth: int,
        used: frozenset[tuple[int, int]],
        goal: str,
    ) -> _Step:
        """Return the best chain of at most ``depth`` SWAPs from this state.

        ``used`` holds the coupling edges the chain has already applied. Reusing
        one can only walk back around a cycle -- the shortest such walk is the
        immediate undo, because a SWAP is its own inverse -- so every edge is
        applied at most once.
        """

        if depth == 0 or not blocked:
            return _Step((), layout, blocked, 0.0)
        candidates = sorted(
            (
                (key(swapped(layout, left, right), blocked, goal), (left, right))
                for left, right in edges
                if (left, right) not in used
            ),
            key=lambda row: (row[0], row[1]),
        )
        if not candidates:
            # The chain has already applied every coupling edge, which happens on a
            # device with fewer edges than the search depth.
            return _Step((), layout, blocked, 0.0)
        best: _Step | None = None
        for _, swap in candidates[: min(search_width, len(candidates))]:
            trial = swapped(layout, *swap)
            order, next_blocked, next_waiting = advance(trial, blocked, waiting)
            # Qiskit's step score: each SWAP costs three operations, and every
            # operation the chain makes executable pays for one. A SWAP that pays
            # for itself is worth taking, so the one-SWAP chain competes against the
            # longer ones rather than being forced to spend the whole depth.
            one = _Step((swap,), trial, next_blocked, float(len(order)) - 3.0)
            if depth == 1:
                candidate = one
            else:
                deeper = search(
                    trial,
                    next_blocked,
                    next_waiting,
                    depth - 1,
                    used | {swap},
                    goal,
                )
                longer = _Step(
                    (swap,) + deeper.chain,
                    deeper.layout,
                    deeper.blocked,
                    float(len(order)) + deeper.score - 3.0,
                )
                candidate = one if one.score >= longer.score else longer
            # ``candidates`` is sorted by the objective, and a strict improvement
            # keeps the best-ranked chain whenever the scores tie.
            if best is None or candidate.score > best.score:
                best = candidate
        assert best is not None
        return best

    layout = placement
    current = tuple(index for index in range(count) if pending[index] == 0)
    waiting = list(pending)
    placements: list[tuple[int, ...]] = [() for _ in range(count)]
    sequence: list[int] = []
    swaps: list[tuple[int, int]] = []
    swap_placements: list[int] = []
    swap_anchors: list[int] = []
    # A chain of distinct coupling edges is not guaranteed to move the layout: the
    # three edges of a triangle act as the identity. The planner therefore records
    # every (layout, front layer) state it has stood on, so a chain that returns to
    # one is visible -- the front layer is part of the state, because the same
    # layout with a shorter front layer is progress, not a cycle. With ``repair``
    # the visible cycle is replaced by the single SWAP the shipped planner would
    # take; without it the cycle is reported. The shipped planner's stall guard is
    # carried alongside so that no objective can spin forever.
    visited: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    stalled = 0
    stalled_limit = STALLED_SWAP_LIMIT_PER_WIRE * n_wires

    while current:
        before = len(sequence)
        order, current, waiting = advance(layout, current, waiting)
        for index in order:
            placements[index] = tuple(layout[wire] for wire in wires[index])
            sequence.append(index)
        if len(sequence) > before:
            stalled = 0
        if not current:
            break
        step = search(layout, current, waiting, search_depth, frozenset(), objective)
        # The shipped planner's objective is allowed to move a wire back, so a
        # layout it has stood on before is not by itself a failure for that
        # objective. Novelty is required of the reference objectives, where
        # returning to a held state is exactly the failure this measures.
        cycled = objective != "sabre" and (step.layout, step.blocked) in visited
        if cycled and repair:
            step = search(layout, current, waiting, 1, frozenset(), "sabre")
        if not step.chain:
            raise ValueError(
                "lookahead found no SWAP chain that unblocks the front layer"
            )
        if cycled and not repair:
            raise ValueError(
                "lookahead has no SWAP chain that unblocks the front layer: its "
                "objective returns the same state it already held, with "
                f"{len(current)} operations still blocked"
            )
        visited.add((layout, current))
        anchor = current[0]
        for swap in step.chain:
            layout = swapped(layout, *swap)
            swaps.append(swap)
            swap_placements.append(len(sequence))
            swap_anchors.append(anchor)
            before = len(sequence)
            order, current, waiting = advance(layout, current, waiting)
            for index in order:
                placements[index] = tuple(layout[wire] for wire in wires[index])
                sequence.append(index)
            if len(sequence) > before:
                stalled = 0
            else:
                stalled += 1
            if stalled > stalled_limit:
                raise ValueError(
                    "lookahead made no progress after "
                    f"{stalled} consecutive swaps on {n_wires} wires"
                )
            if not current:
                break

    if len(sequence) != count:
        raise RuntimeError(
            f"lookahead placed {len(sequence)} of {count} instructions; the front "
            "layer emptied before the program was fully placed"
        )
    replay = list(placement)
    for left, right in swaps:
        for logical in range(n_wires):
            if replay[logical] == left:
                replay[logical] = right
            elif replay[logical] == right:
                replay[logical] = left
    if tuple(replay) != layout:
        raise RuntimeError("lookahead plan does not replay to its planned layout")
    return SabrePlan(
        sequence=tuple(sequence),
        placements=tuple(placements),
        swaps=tuple(swaps),
        swap_placements=tuple(swap_placements),
        swap_anchors=tuple(swap_anchors),
        initial_logical_to_physical=placement,
        final_logical_to_physical=layout,
    )


@contextmanager
def _reference_planner(planner: Any) -> Iterator[None]:
    """Route with ``planner`` in place of the shipped SWAP search.

    ``routing`` binds the shipped search at import, and ``sabre`` binds the one
    its layout search calls, so both bindings are replaced: a measured plan must
    be materialized by the shipped consumer, not by a copy of it.
    """

    with (
        patch.object(routing_module, "plan_sabre_swaps", planner),
        patch.object(sabre_module, "plan_sabre_swaps", planner),
    ):
        yield


def _routed(
    program: CircuitIR, device: CouplingMap, configuration: Configuration
) -> Any:
    """Compile one program through the consumer of the configured planner."""

    planner = configuration.planner()
    with _reference_planner(planner) if planner else nullcontext():
        routed = route_to_topology(program, device, strategy=configuration.strategy)
    return record_post_routing_optimization(optimize(routed))


def _state(program: CircuitIR) -> torch.Tensor:
    return fq.Circuit.from_ir(program).state()


def _inserted_swaps(compiled: CircuitIR) -> tuple[tuple[int, int], ...]:
    """Return the SWAPs a route inserted, in the order the router emitted them."""

    return tuple(
        instruction.wires
        for instruction in compiled
        if instruction.name == "swap"
        and instruction.metadata.get("routing_phase") is not None
    )


def run_benchmark(
    *,
    topologies: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = SEEDS,
    configurations: tuple[Configuration, ...] = DEFAULT_CONFIGURATIONS,
    verify_states: bool = True,
) -> dict[str, Any]:
    """Measure every configuration against the shipped strategies.

    Each configuration is compiled through ``route_to_topology`` and the shipped
    optimizer, so the retained SWAP count is the one the product reports and the
    state comparison is a real end-to-end check. Implementations are replaced by
    patching the two bindings of the shipped SWAP search, which is the same
    replacement mechanism the routing conformance suite uses.

    Args:
        topologies: Topology names to measure; every declared topology by default.
        seeds: Program seeds to measure.
        configurations: Planners to measure; the shipped strategies should be
            included, because they are the baseline every ratio is taken against.
        verify_states: Compare the compiled state of every program against its
            source state. Disabling it drops the numerical evidence, so only do so
            when the same planner has already been verified elsewhere.

    Returns:
        A JSON-serializable payload: the measured basis, each configuration's swap
        counts and per-case record against the shipped ``sabre_layout`` strategy,
        and the fidelity anchor for the search machinery.
    """

    cases = case_basis(topologies=topologies, seeds=seeds)
    labels = [configuration.label for configuration in configurations]
    failures: dict[str, list[str]] = {label: [] for label in labels}
    planned: dict[str, int] = dict.fromkeys(labels, 0)
    retained: dict[str, int] = dict.fromkeys(labels, 0)
    instructions: dict[str, int] = dict.fromkeys(labels, 0)
    illegal: dict[str, int] = dict.fromkeys(labels, 0)
    better: dict[str, int] = dict.fromkeys(labels, 0)
    worse: dict[str, int] = dict.fromkeys(labels, 0)
    tie: dict[str, int] = dict.fromkeys(labels, 0)
    seconds: dict[str, float] = dict.fromkeys(labels, 0.0)
    verified = 0
    max_state_difference = 0.0
    anchored_cases = 0
    anchor_identical_swaps = 0
    anchor_equal_planned_swaps = 0
    case_records: list[dict[str, Any]] = []

    for case in cases:
        device = case.device()
        program = case.program()
        source_state = _state(program) if verify_states else None
        measured: dict[str, tuple[int, tuple[tuple[int, int], ...], int]] = {}
        for configuration in configurations:
            label = configuration.label
            started = time.perf_counter()
            try:
                compiled = _routed(program, device, configuration)
            except ValueError as error:
                failures[label].append(f"{case.label}: {error}")
                seconds[label] += time.perf_counter() - started
                continue
            seconds[label] += time.perf_counter() - started
            routing = compiled.metadata["routing"]
            planned[label] += routing["inserted_swap_count"]
            value = routing["post_optimization_inserted_swap_count"]
            retained[label] += value
            instructions[label] += len(compiled)
            illegal[label] += sum(
                1
                for instruction in compiled
                if len(instruction.wires) == 2
                and not device.has_edge(*instruction.wires)
            )
            measured[label] = (
                value,
                _inserted_swaps(compiled),
                routing["inserted_swap_count"],
            )
            if source_state is not None:
                difference = float(
                    torch.max(torch.abs(_state(compiled) - source_state)).item()
                )
                max_state_difference = max(max_state_difference, difference)
                verified += 1
        # The shipped strategies are measured first, so every reference row in this
        # case has its baseline available. Only cases the configuration planned are
        # counted, so a configuration that fails closed on some cases is never
        # compared on output it did not produce.
        baseline = measured.get("sabre_layout")
        shipped = measured.get("sabre")
        anchor = measured.get("sabre/d1w1")
        if baseline is not None:
            for label, (value, _, _) in measured.items():
                if label == "sabre_layout":
                    continue
                if value < baseline[0]:
                    better[label] += 1
                elif value == baseline[0]:
                    tie[label] += 1
                else:
                    worse[label] += 1
        if shipped is not None and anchor is not None:
            anchored_cases += 1
            if anchor[1] == shipped[1]:
                anchor_identical_swaps += 1
            if anchor[2] == shipped[2]:
                anchor_equal_planned_swaps += 1
        # One row per (case, configuration), so a per-case claim in the recorded
        # result can be re-checked rather than taken on trust.
        for configuration in configurations:
            label = configuration.label
            record: dict[str, Any] = {
                "case": case.label,
                "topology": case.topology,
                "configuration": label,
                "failed": label not in measured,
            }
            if label in measured:
                record["retained_inserted_swap_count"] = measured[label][0]
                record["planned_inserted_swap_count"] = measured[label][2]
            case_records.append(record)

    rows = []
    layout_retained = retained.get("sabre_layout", 0)
    sabre_retained = retained.get("sabre", 0)
    for configuration in configurations:
        label = configuration.label
        value = retained[label]
        rows.append(
            {
                "label": label,
                "strategy": configuration.strategy,
                "objective": configuration.objective or "shipped",
                "search_depth": configuration.search_depth,
                "search_width": configuration.search_width,
                "repair": configuration.repair,
                "planned_case_count": len(cases) - len(failures[label]),
                "failed_case_count": len(failures[label]),
                "failed_cases": failures[label],
                "off_device_two_wire_instruction_count": illegal[label],
                "planned_inserted_swap_count": planned[label],
                "retained_inserted_swap_count": value,
                "instruction_count": instructions[label],
                "ratio_to_sabre": (value / sabre_retained) if sabre_retained else 0.0,
                "ratio_to_sabre_layout": (
                    (value / layout_retained) if layout_retained else 0.0
                ),
                "better_than_sabre_layout_case_count": better[label],
                "worse_than_sabre_layout_case_count": worse[label],
                "tie_with_sabre_layout_case_count": tie[label],
                "planning_seconds": seconds[label],
            }
        )
    return {
        "schema": "flagquantum_compiler_lookahead_swap_benchmark_v1",
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "reference_algorithm": "qiskit_lookahead_swap_beam_search",
        "cost_model": (
            "instruction count of the compiled program after topology routing and "
            "the shipped optimizer; swaps are counted by "
            "routing.inserted_swap_count and routing.post_optimization_inserted_swap_count"
        ),
        "case_count": len(cases),
        "topologies": list(topologies if topologies is not None else _TOPOLOGIES),
        "seeds": list(seeds),
        "gate_factors": list(GATE_FACTORS),
        "shipped": {
            label: {
                "planned_inserted_swap_count": planned[label],
                "retained_inserted_swap_count": retained[label],
                "instruction_count": instructions[label],
            }
            for label in ("sabre", "sabre_layout")
            if label in failures
        },
        "configurations": rows,
        "case_records": case_records,
        "verified_case_count": verified,
        "max_state_difference": max_state_difference,
        "fidelity_anchor": {
            "configuration": "sabre/d1w1",
            "case_count": anchored_cases,
            "identical_inserted_swap_sequence_case_count": anchor_identical_swaps,
            "equal_planned_swap_count_case_count": anchor_equal_planned_swaps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--topology",
        action="append",
        choices=tuple(_TOPOLOGIES),
        help="topology to measure; repeat to select several",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="program seed to measure; repeat to select several",
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(c.label for c in DEFAULT_CONFIGURATIONS),
        help="configuration to measure; repeat to select several",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="skip the compiled-versus-source state comparison",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    selected = (
        DEFAULT_CONFIGURATIONS
        if args.config is None
        else tuple(c for c in DEFAULT_CONFIGURATIONS if c.label in set(args.config))
    )
    payload = run_benchmark(
        topologies=None if args.topology is None else tuple(args.topology),
        seeds=SEEDS if args.seed is None else tuple(args.seed),
        configurations=selected,
        verify_states=not args.skip_verification,
    )
    shipped = payload["shipped"]["sabre_layout"]["retained_inserted_swap_count"]
    print(
        f"{payload['case_count']} cases on {', '.join(payload['topologies'])}; "
        f"sabre_layout retains {shipped} inserted SWAPs"
    )
    print(f"{'configuration':24s} {'retained':>9s} {'vs layout':>10s} {'failed':>7s}")
    for row in payload["configurations"]:
        value = row["retained_inserted_swap_count"]
        ratio = value / shipped if shipped else 0.0
        print(
            f"{row['label']:24s} {value:9d} {ratio:9.3f}x "
            f"{row['failed_case_count']:7d}"
        )
    print(
        f"verified {payload['verified_case_count']} compiled programs; "
        f"max state difference {payload['max_state_difference']:.3e}"
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
