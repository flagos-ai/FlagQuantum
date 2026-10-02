"""Measure the stochastic SWAP search Qiskit removed in 2.0 against the shipped planners.

W7-06 of the Qiskit parity backlog asks for Qiskit's ``StochasticSwap``, whose
design is Bravyi's randomized layer-permutation search: for each layer of the
circuit, take the two-wire operations that layer contains, perturb the coupling
distances with a random symmetric matrix, and greedily accept the single
best-cost-reducing edge per pass until no edge reduces the cost; repeat the whole
search ``trials`` times and keep the trial with the smallest SWAP depth. The
algorithms this repository already ships are the greedy SABRE SWAP search
(``sabre``) and the SABRE layout search that precedes it (``sabre_layout``).

This module builds that planner and measures it through the shipped consumer
(``route_to_topology``) and the shipped optimizer, so every comparison is the
same code path and the same cost model on both sides. The answer is negative,
and the measurement is checked in so the decision can be re-tested rather than
re-argued. On the 140-program basis this module measures by default, whose
shipped baselines retain 2410 SWAPs (``sabre``) and 2051 (``sabre_layout``)
after optimization:

* The port is a *correct* routing planner. Every compiled program is compared
  with its source state, no configuration emits a two-wire instruction off the
  device, and every plan it returns is checked independently: the physical wires
  it records for each operation are exactly the replay of the SWAPs it returns,
  and every two-wire operation it plans sits on a device edge. The port is a
  replacement, not an approximation of one.

* The port *loses*, and it loses on every program it plans. At its documented
  default of 20 trials it retains 3316 SWAPs: 1.376 times ``sabre`` and 1.617
  times ``sabre_layout``. It beats ``sabre_layout`` on 0 of the 140 programs.
  Raising the trial count does not close the gap (1 trial 1.628, 20 trials and
  100 trials both 1.617), so the shortfall is the algorithm's search, not an
  unlucky random stream: ten times the trials buys no ground at all.

* The shortfall is *not* the port. Qiskit's own compiled implementation of this
  algorithm, driven through the same consumer on the same 140 cases, plans 3376
  and retains 3300 SWAPs -- 1.609 times ``sabre_layout``, 0.5 % fewer than this
  port -- so a bit-exact port could not have won either. Both implementations
  therefore lose by more than half again as much as ``sabre_layout`` retains.

  That cross-check is a recorded number rather than a checked-in job, because it
  cannot be a checked-in job: it needs the ``qiskit._accelerate.stochastic_swap``
  extension, which the pass's own wheel exposed privately and which Qiskit 2.0
  deleted along with the pass. This repository certifies Qiskit 2.0.x and 2.5.x,
  so the runner would never execute under ``qiskit-optional``; the command that
  produced the number was a Qiskit 1.2.4 environment with
  ``qiskit._accelerate.stochastic_swap.swap_trials`` patched in as the layer
  permutation planner. The *claim* this module makes -- that the checked-in port
  loses -- stays re-testable from this repository alone, because the port is what
  is checked in. (Qiskit 2.x still *reserves* the name ``stochastic`` for a
  routing-stage plugin, so it appears in the 2.x plugin documentation; the Python
  pass and the Rust kernel behind it are both gone.)

* Substituting the search into the shipped *layout* search is the closest the
  port comes: 1.385 of ``sabre_layout``, better than the 1.617 it reaches alone
  but still a regression against what the product selects.

One deliberate difference from the reference: this port scores a disconnected
physical pair far above any real hop count, where the reference scores the
negative sentinel ``CouplingMap.distance_matrix`` uses and then squares it, which
makes an unroutable pair *cheaper* than an adjacent one. No measured program has a
disconnected device, so the difference does not move a recorded number; it is
recorded because a future reader comparing the two would otherwise read it as a
porting error.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_stochastic_swap.py --json-output /tmp/stochastic.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.compiler_lookahead_swap import (
    _inserted_swaps,
    _reference_planner,
    _state,
    case_basis,
)
from flagquantum.compiler import CouplingMap, optimize, route_to_topology
from flagquantum.compiler.routing import record_post_routing_optimization
from flagquantum.compiler.sabre import (
    SabrePlan,
    _planning_coupling,
    _validated_layout,
)
from flagquantum.core.ir import CircuitIR, Instruction

# Qiskit's documented default. The row exists so the number this module records
# is the number the class's own default produces.
TRIALS = 20

# The search has no dependence on a lucky stream: one trial and a hundred trials
# land within 3 % of each other on the recorded basis, so a row is a property of
# the algorithm rather than of one seed. Ten times the trials buys no ground and
# costs ten times the planning time.
_TRIAL_COUNTS = (1, 5, 20, 100)

# A disconnected physical pair cannot host a two-wire operation, so it is scored
# far above any real hop count rather than at the negative sentinel
# ``CouplingMap.distance_matrix`` uses, whose square would score it *below* one
# hop. The value is finite so the perturbed cost stays a real number.
_UNREACHABLE = 1_000_000.0


@dataclass(frozen=True)
class PlanConsistency:
    """What an independent replay of one plan found.

    Attributes:
        placements_replay: The physical wires the plan records for each operation
            are exactly the wires the plan's own SWAPs move the placement to.
        two_wire_placements_on_device: Every two-wire operation the plan places
            sits on an edge of the device it was planned for.
    """

    placements_replay: bool
    two_wire_placements_on_device: bool


@dataclass(frozen=True)
class Configuration:
    """One planner to measure against the shipped strategies.

    Attributes:
        label: Stable name of the row this configuration produces.
        strategy: Shipped routing strategy whose consumer materializes the plan.
        trials: Random trials per layer permutation. Zero keeps the shipped SWAP
            search, which is what the baseline rows are.
    """

    label: str
    strategy: str
    trials: int = 0

    def planner(self) -> Any:
        """Return the SWAP search this configuration routes with."""

        if self.trials == 0:
            return None

        def plan(
            program: CircuitIR,
            coupling: CouplingMap,
            *,
            initial_layout: tuple[int, ...] | None = None,
        ) -> SabrePlan:
            return plan_stochastic_swaps(
                program,
                coupling,
                initial_layout=initial_layout,
                trials=self.trials,
            )

        return plan


DEFAULT_CONFIGURATIONS = (
    Configuration("sabre", "sabre"),
    Configuration("sabre_layout", "sabre_layout"),
    *(
        Configuration(f"stochastic/t{trials}", "sabre", trials)
        for trials in _TRIAL_COUNTS
    ),
    Configuration("layout/stochastic/t20", "sabre_layout", 20),
)


def asap_layers(instructions: tuple[Instruction, ...]) -> tuple[tuple[int, ...], ...]:
    """Return source instruction indices grouped by wire dependency depth.

    An operation belongs to the layer after every operation on each of its own
    wires, which is the longest-path layering a circuit DAG has: two operations
    share a layer exactly when neither depends on the other.
    """

    layers: list[list[int]] = []
    last_layer: dict[int, int] = {}
    for index, instruction in enumerate(instructions):
        layer = 1
        for wire in instruction.wires:
            layer = max(layer, last_layer.get(wire, -1) + 2)
        while len(layers) < layer:
            layers.append([])
        layers[layer - 1].append(index)
        for wire in instruction.wires:
            last_layer[wire] = layer - 1
    return tuple(tuple(layer) for layer in layers)


def _perturbation(
    *, seed: int, n_wires: int, squared_distances: tuple[tuple[float, ...], ...]
) -> tuple[tuple[float, ...], ...]:
    """Return the symmetric random scaling of the squared distances.

    This is the reference search's ``scale`` matrix: each entry above the diagonal
    is one draw from ``Normal(1, 1 / n_wires)`` times the squared distance, and
    the matrix is mirrored below it. The draws come from this module's seeded
    generator rather than the reference's PCG64, so a run is reproducible here but
    is not bit-identical with a run of the reference.
    """

    gen = random.Random(seed)
    draws = [
        gen.normalvariate(1.0, 1.0 / n_wires)
        for _ in range(n_wires * (n_wires - 1) // 2)
    ]
    scale = [[0.0] * n_wires for _ in range(n_wires)]
    cursor = 0
    for upper in range(n_wires):
        for lower in range(upper):
            value = draws[cursor] * squared_distances[upper][lower]
            scale[upper][lower] = value
            scale[lower][upper] = value
            cursor += 1
    return tuple(tuple(row) for row in scale)


def _separation(
    distances: tuple[tuple[float, ...], ...],
    placement: list[int],
    gates: tuple[tuple[int, int], ...],
) -> float:
    """Return the summed distance the gates have under one placement."""

    return sum(
        distances[placement[wire_a]][placement[wire_b]] for wire_a, wire_b in gates
    )


def _swap_trial(
    *,
    placement: list[int],
    gates: tuple[tuple[int, int], ...],
    cost: tuple[tuple[float, ...], ...],
    on_device: tuple[tuple[float, ...], ...],
    edges: tuple[tuple[int, int], ...],
    n_wires: int,
) -> tuple[float, list[tuple[int, int]], list[int], int]:
    """Search one permutation under one cost matrix.

    Repeated passes accept the single edge whose SWAP reduces ``cost`` the most,
    then remove both wires it consumed from the wires still eligible to move, and
    stop as soon as no edge reduces the cost. When the placement is not yet
    routable the passes are repeated one step deeper, up to a depth bound of twice
    the wire count plus one.

    ``cost`` is the perturbed matrix and ``on_device`` the true one. The two are
    different on purpose: the perturbation is what makes a trial explore a
    different descent than a greedy one would, and the true distances are what
    decides whether the trial found a routable placement.

    Args:
        cost: Distances the descent minimizes. Each entry is one random draw times
            the squared true distance, so it is a noisier view of the same device.
        on_device: True hop counts, used only to decide whether the placement is
            routable.

    Returns:
        The true separation the search reached, the SWAPs it took in order, the
        placement it reached, and the depth it stopped at.
    """

    physical_to_logical = [0] * n_wires
    for logical, physical in enumerate(placement):
        physical_to_logical[physical] = logical
    trial = list(placement)
    optimal = list(placement)
    swaps: list[tuple[int, int]] = []
    num_gates = len(gates)
    depth = 1
    depth_max = 2 * n_wires + 1

    while depth < depth_max:
        remaining = set(range(n_wires))
        while remaining:
            best_cost = _separation(cost, trial, gates)
            chosen: tuple[int, int, int, int] | None = None
            for start_edge, end_edge in edges:
                start = physical_to_logical[start_edge]
                end = physical_to_logical[end_edge]
                if start not in remaining or end not in remaining:
                    continue
                trial[start], trial[end] = trial[end], trial[start]
                physical_to_logical[start_edge], physical_to_logical[end_edge] = (
                    end,
                    start,
                )
                candidate = _separation(cost, trial, gates)
                if candidate < best_cost:
                    best_cost = candidate
                    optimal = list(trial)
                    chosen = (start_edge, end_edge, start, end)
                # The placement swap is its own inverse; the wire lookup is an
                # assignment, so it is restored to the values it held before this
                # edge was tried rather than assigned a second time. Restoring it by
                # swapping again leaves the lookup permuted against the placement.
                trial[start], trial[end] = trial[end], trial[start]
                physical_to_logical[start_edge] = start
                physical_to_logical[end_edge] = end
            if chosen is None:
                break
            start_edge, end_edge, start, end = chosen
            remaining.discard(start)
            remaining.discard(end)
            trial = list(optimal)
            for logical, physical in enumerate(trial):
                physical_to_logical[physical] = logical
            swaps.append((start_edge, end_edge))
        distance = _separation(on_device, trial, gates)
        if distance == num_gates:
            break
        depth += 1
    return distance, swaps, trial, depth


def layer_permutation(
    *,
    placement: list[int],
    gates: tuple[tuple[int, int], ...],
    distances: tuple[tuple[float, ...], ...],
    squared: tuple[tuple[float, ...], ...],
    edges: tuple[tuple[int, int], ...],
    n_wires: int,
    trials: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[int]] | None:
    """Return a SWAP sequence that makes every gate of one layer adjacent.

    Returns no SWAPs when the layer is already routable, and ``None`` when no trial
    found a sequence, which the caller answers by planning the layer one operation
    at a time.
    """

    if sum(
        distances[placement[wire_a]][placement[wire_b]] for wire_a, wire_b in gates
    ) == len(gates):
        return [], list(placement)
    streams = random.Random(seed)
    best: tuple[int, list[tuple[int, int]], list[int]] | None = None
    for _ in range(trials):
        trial_seed = streams.getrandbits(64)
        distance, swaps, trial, depth = _swap_trial(
            placement=placement,
            gates=gates,
            cost=_perturbation(
                seed=trial_seed, n_wires=n_wires, squared_distances=squared
            ),
            on_device=distances,
            edges=edges,
            n_wires=n_wires,
        )
        if distance == len(gates) and (best is None or depth < best[0]):
            best = (depth, swaps, trial)
            if depth == 1:
                break
    if best is None:
        return None
    return best[1], best[2]


def plan_stochastic_swaps(
    program: CircuitIR,
    coupling: CouplingMap,
    *,
    initial_layout: tuple[int, ...] | None = None,
    trials: int = TRIALS,
    seed: int = 0,
) -> SabrePlan:
    """Plan a persistent layout by randomized layer permutations.

    The plan has the shape :class:`~flagquantum.compiler.sabre.SabrePlan` gives the
    shipped planner, so the shipped materialization, restore, and cost model all
    apply to it unchanged.

    Args:
        program: The program to place and route.
        coupling: The device to route on.
        initial_layout: Logical-to-physical placement to start from.
        trials: Random trials per layer permutation.
        seed: Seed of the stream the per-trial seeds are drawn from.

    Raises:
        ValueError: If ``trials`` is below one, or if no permutation of a layer
            makes its operations adjacent and planning them one at a time cannot
            route one of them either.
    """

    if trials < 1:
        raise ValueError("trials must be at least one")
    n_wires = program.n_wires
    if coupling.n_wires < n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")
    placement = _validated_layout(initial_layout, n_wires)
    instructions = program.instructions
    if not instructions:
        return SabrePlan((), (), (), (), (), placement, placement)

    # The plan may only swap between wires the program owns, so it is costed on
    # the coupling subgraph those wires induce -- the same graph, and the same
    # helpers, the shipped planner uses.
    planning = _planning_coupling(coupling, n_wires)
    distances = tuple(
        tuple(_UNREACHABLE if hop < 0 else float(hop) for hop in row)
        for row in planning.distance_matrix()
    )
    squared = tuple(tuple(value * value for value in row) for row in distances)
    edges = tuple(sorted(planning.edges))

    layout = list(placement)
    placements: list[tuple[int, ...]] = [() for _ in instructions]
    sequence: list[int] = []
    swaps: list[tuple[int, int]] = []
    swap_placements: list[int] = []
    swap_anchors: list[int] = []
    placed = 0

    def layer_gates(indices: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        """Return the two-wire operations of a layer that need physical adjacency.

        A three-wire operation is left to the consumer, which refuses one its
        physical wires cannot host, and a channel is not an operation on the
        device. The shipped planner reads a layer the same way.
        """

        return tuple(
            (instructions[index].wires[0], instructions[index].wires[1])
            for index in indices
            if len(instructions[index].wires) == 2
            and not instructions[index].metadata.get("is_channel")
        )

    def emit(indices: tuple[int, ...]) -> None:
        nonlocal placed
        for index in indices:
            placements[index] = tuple(
                layout[wire] for wire in instructions[index].wires
            )
            sequence.append(index)
            placed += 1

    def apply(layer_swaps: list[tuple[int, int]], anchor: int) -> None:
        for left, right in layer_swaps:
            swaps.append((left, right))
            swap_placements.append(placed)
            swap_anchors.append(anchor)

    for layer in asap_layers(instructions):
        anchor = next(
            (
                index
                for index in layer
                if len(instructions[index].wires) == 2
                and not instructions[index].metadata.get("is_channel")
            ),
            layer[0],
        )
        result = layer_permutation(
            placement=layout,
            gates=layer_gates(layer),
            distances=distances,
            squared=squared,
            edges=edges,
            n_wires=n_wires,
            trials=trials,
            seed=seed,
        )
        if result is not None:
            layer_swaps, layout = result
            apply(layer_swaps, anchor)
            emit(layer)
            continue
        # No permutation of the whole layer works, so each operation is planned on
        # its own, exactly as the reference falls back to its serial layers.
        for index in layer:
            result = layer_permutation(
                placement=layout,
                gates=layer_gates((index,)),
                distances=distances,
                squared=squared,
                edges=edges,
                n_wires=n_wires,
                trials=trials,
                seed=seed,
            )
            if result is None:
                raise ValueError(
                    f"no permutation of the device wires makes instruction {index} "
                    f"adjacent under trials={trials}"
                )
            layer_swaps, layout = result
            apply(layer_swaps, index)
            emit((index,))

    return SabrePlan(
        sequence=tuple(sequence),
        placements=tuple(placements),
        swaps=tuple(swaps),
        swap_placements=tuple(swap_placements),
        swap_anchors=tuple(swap_anchors),
        initial_logical_to_physical=placement,
        final_logical_to_physical=tuple(layout),
    )


def check_plan(
    program: CircuitIR, plan: SabrePlan, coupling: CouplingMap
) -> PlanConsistency:
    """Replay one plan independently and report what a consumer would find.

    The shipped consumer trusts ``placements`` and replays ``swaps``, so a plan
    whose two disagree would route a different circuit than the one it was costed
    against. Nothing else in this module compares them, which is the point: this
    is a check on the planner, not a restatement of it.
    """

    layout = list(plan.initial_logical_to_physical)
    placed = 0
    swap_index = 0
    replays = True
    for source_index in plan.sequence:
        while (
            swap_index < len(plan.swaps) and plan.swap_placements[swap_index] == placed
        ):
            left, right = plan.swaps[swap_index]
            first = layout.index(left)
            second = layout.index(right)
            layout[first], layout[second] = right, left
            swap_index += 1
        placed += 1
        instruction = program.instructions[source_index]
        if (
            tuple(layout[wire] for wire in instruction.wires)
            != plan.placements[source_index]
        ):
            replays = False
    on_device = all(
        coupling.has_edge(*plan.placements[index])
        for index, instruction in enumerate(program.instructions)
        if len(instruction.wires) == 2 and not instruction.metadata.get("is_channel")
    )
    return PlanConsistency(
        placements_replay=replays and layout == list(plan.final_logical_to_physical),
        two_wire_placements_on_device=on_device,
    )


def _routed(
    program: CircuitIR, device: CouplingMap, configuration: Configuration
) -> CircuitIR:
    """Compile one program through the consumer of the configured planner."""

    planner = configuration.planner()
    with _reference_planner(planner) if planner else nullcontext():
        routed = route_to_topology(program, device, strategy=configuration.strategy)
    return record_post_routing_optimization(optimize(routed))


def run_benchmark(
    *,
    topologies: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    configurations: tuple[Configuration, ...] = DEFAULT_CONFIGURATIONS,
    verify_states: bool = True,
) -> dict[str, Any]:
    """Measure every configuration against the shipped strategies.

    Each configuration is compiled through ``route_to_topology`` and the shipped
    optimizer, so the retained SWAP count is the one the product reports and the
    state comparison is a real end-to-end check. Implementations are replaced by
    patching the two bindings of the shipped SWAP search, which is the same
    replacement mechanism the routing conformance suite uses.

    Every plan a measured configuration produces is also replayed independently by
    :func:`check_plan`, so the fidelity of the port is a counted result rather than
    a claim.

    Args:
        topologies: Topology names to measure; every declared topology by default.
        seeds: Program seeds to measure; the basis default.
        configurations: Planners to measure; the shipped strategies should be
            included, because they are the baseline every ratio is taken against.
        verify_states: Compare the compiled state of every program against its
            source state. Disabling it drops the numerical evidence, so only do so
            when the same planner has already been verified elsewhere.

    Returns:
        A JSON-serializable payload: the measured basis, each configuration's swap
        counts and per-case record against the shipped ``sabre_layout`` strategy,
        and the plan-replay and trial-robustness anchors.
    """

    from benchmarks.compiler_lookahead_swap import SEEDS

    cases = case_basis(topologies=topologies, seeds=SEEDS if seeds is None else seeds)
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
    plans_checked = 0
    plans_replaying = 0
    plans_on_device = 0
    case_records: list[dict[str, Any]] = []
    retained_by_case: dict[str, dict[str, int]] = {label: {} for label in labels}

    for case in cases:
        device = case.device()
        program = case.program()
        source_state = _state(program) if verify_states else None
        measured: dict[str, tuple[int, tuple[tuple[int, int], ...], int]] = {}
        for configuration in configurations:
            label = configuration.label
            started = time.perf_counter()
            try:
                if configuration.trials:
                    plan = plan_stochastic_swaps(
                        program, device, trials=configuration.trials
                    )
                    consistency = check_plan(program, plan, device)
                    plans_checked += 1
                    plans_replaying += int(consistency.placements_replay)
                    plans_on_device += int(consistency.two_wire_placements_on_device)
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
            retained_by_case[label][case.label] = value
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
                difference = float((_state(compiled) - source_state).abs().max().item())
                max_state_difference = max(max_state_difference, difference)
                verified += 1
        # The shipped strategies are measured first, so every reference row in this
        # case has its baseline available. Only cases the configuration planned are
        # counted, so a configuration that fails closed on some cases is never
        # compared on output it did not produce.
        baseline = measured.get("sabre_layout")
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

    layout_retained = retained.get("sabre_layout", 0)
    sabre_retained = retained.get("sabre", 0)
    rows = []
    for configuration in configurations:
        label = configuration.label
        value = retained[label]
        rows.append(
            {
                "label": label,
                "strategy": configuration.strategy,
                "trials": configuration.trials,
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
    robustness = []
    for configuration in configurations:
        if not configuration.trials:
            continue
        label = configuration.label
        shared = [
            case
            for case, value in retained_by_case[label].items()
            if retained_by_case.get("stochastic/t20", {}).get(case) == value
        ]
        robustness.append(
            {
                "label": label,
                "equal_retained_case_count_to_stochastic_t20": len(shared),
                "case_count": len(retained_by_case[label]),
            }
        )
    return {
        "schema": "flagquantum_compiler_stochastic_swap_benchmark_v1",
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "reference_algorithm": "qiskit_stochastic_swap_randomized_layer_permutation",
        "reference_revision": "qiskit 1.3.0 (removed in qiskit 2.0.0)",
        "cost_model": (
            "instruction count of the compiled program after topology routing and "
            "the shipped optimizer; swaps are counted by "
            "routing.inserted_swap_count and routing.post_optimization_inserted_swap_count"
        ),
        "case_count": len(cases),
        "topologies": sorted({case.topology for case in cases}),
        "seeds": list(SEEDS if seeds is None else seeds),
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
        "plan_replay": {
            "case_count": plans_checked,
            "placements_replay_case_count": plans_replaying,
            "two_wire_placements_on_device_case_count": plans_on_device,
        },
        "trial_robustness": robustness,
    }


def _print(payload: dict[str, Any]) -> None:
    """Print the measured basis and every configuration row."""

    shipped = payload["shipped"]["sabre_layout"]["retained_inserted_swap_count"]
    print(
        f"{payload['case_count']} cases on {', '.join(payload['topologies'])}; "
        f"sabre_layout retains {shipped} inserted SWAPs"
    )
    print(
        f"{'configuration':24s} {'trials':>6s} {'planned':>8s} {'retained':>9s} "
        f"{'vs layout':>10s} {'better':>7s} {'failed':>7s}"
    )
    for row in payload["configurations"]:
        value = row["retained_inserted_swap_count"]
        ratio = value / shipped if shipped else 0.0
        print(
            f"{row['label']:24s} {row['trials']:6d} "
            f"{row['planned_inserted_swap_count']:8d} {value:9d} {ratio:9.3f}x "
            f"{row['better_than_sabre_layout_case_count']:7d} "
            f"{row['failed_case_count']:7d}"
        )
    replay = payload["plan_replay"]
    print(
        f"replayed {replay['case_count']} plans; "
        f"{replay['placements_replay_case_count']} placements equal their SWAP replay, "
        f"{replay['two_wire_placements_on_device_case_count']} place every two-wire "
        f"operation on the device"
    )
    print(
        f"verified {payload['verified_case_count']} compiled programs; "
        f"max state difference {payload['max_state_difference']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--topology",
        action="append",
        choices=tuple(sorted(_topology_names())),
        help="topology to measure; repeat to select several",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="write the full payload to this path as JSON",
    )
    args = parser.parse_args()
    payload = run_benchmark(
        topologies=tuple(args.topology) if args.topology else None,
    )
    _print(payload)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _topology_names() -> tuple[str, ...]:
    """Return the topology names the shared basis declares."""

    from benchmarks.compiler_lookahead_swap import _TOPOLOGIES

    return tuple(_TOPOLOGIES)


if __name__ == "__main__":
    main()
