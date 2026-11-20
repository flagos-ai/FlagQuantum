"""Estimate routing costs and select a deployable strategy before emission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ...core.ir import CircuitIR, ensure_circuit_ir
from .routing_cost import (
    _count_sabre_orientation_swaps,
    _is_routable_two_wire_gate,
    _PricedDevice,
    _source_routing_counts,
    _topology_cost_model,
    _TopologyCostModel,
    require_weight_reading_strategy,
)
from .sabre import (
    SabrePlan,
    _plan_sabre_trial,
    _select_sabre_trials,
    plan_restore_swaps,
    plan_sabre_layout,
    plan_sabre_swaps,
)

# The routing strategies a caller may request. This is the authoritative
# vocabulary: the `program_compilation` limitations in `capability-maturity.toml`
# are the user-facing statement of the same boundary, and
# `tests/team/compiler/test_routing_strategy_documentation.py` fails when the two
# disagree, so a strategy cannot be added here alone.
ROUTING_STRATEGIES = (
    "restore_after_each_gate",
    "persistent_layout",
    "sabre",
    "sabre_layout",
)

# The two strategies that plan SWAPs from a search rather than estimating them.
SABRE_ROUTING_STRATEGIES = ("sabre", "sabre_layout")

# The strategies whose plan the deployment routing contract accepts, which is the
# candidate set an automatic choice may pick from: `auto` must not hand a caller a
# program that fail-closed deployment then refuses. `sabre_layout` is the one
# strategy outside this set, and the reason is structural rather than a ranking: it
# starts from a layout `plan_sabre_layout` searched for, so its
# `initial_logical_to_physical` is the identity on only 23 of the 60 programs in
# `tests/team/compiler/test_routing_conformance.py`, while
# `flagquantum/deployment/routing_evidence.py` requires the identity because a
# deployment artifact must state a physical starting assignment a provider can
# reproduce. It stays a caller-selected strategy and stays priceable, so a caller
# whose provider accepts a permuted start can still ask for it by name.
# `flagquantum/deployment/routing_evidence.py` reads this constant, so the
# accepted set and the selectable set cannot drift apart.
DEPLOYABLE_ROUTING_STRATEGIES = (
    "restore_after_each_gate",
    "persistent_layout",
    "sabre",
)

# The strategies that price a device's declared link error, and so the only ones a
# calibrated device may be routed by. The two shortest-path strategies take their
# route from the cost model's `path`, which is the single place a route is chosen
# for them, so they read a calibration as soon as the model carries one. The two
# SABRE strategies take their route from `plan_sabre_swaps`, which reads a
# `CouplingMap` and minimises a restored SWAP count beside a decayed depth
# heuristic; it has no argument for a per-link weight, so a calibrated device
# would be routed by a planner that never looked at the calibration. Refusing that
# pairing is what keeps a weight from being declared and then ignored.
LINK_ERROR_ROUTING_STRATEGIES = (
    "restore_after_each_gate",
    "persistent_layout",
)


@dataclass(frozen=True)
class RoutingCostEstimate:
    """Lightweight routing size estimate without materializing instructions.

    ``direction_swap_count`` is the part of ``planned_inserted_swap_count`` an
    undirected device cannot produce: it counts the hops inserted only because a
    gate's operands run against the direction its link is declared in. It is zero
    on an undirected device, where no link has a direction to run against, and
    ``direction_semantics`` states which of the two devices the estimate was taken
    against.

    ``expected_link_faults`` is the estimate taken against a device that states a
    per-link error rate, and it is the one quantity here that is not a count: the
    summed two-qubit error probability of every operation the strategy would emit,
    which is the expected number of fault events in the routed program. It is
    ``None`` on a device that declares only its couplings, because there is no such
    quantity to report and ``0.0`` would read as a faultless program.
    """

    strategy: str
    source_instruction_count: int
    estimated_instruction_count: int
    topology_gate_count: int
    routed_gate_count: int
    planned_inserted_swap_count: int
    skipped_channel_count: int
    pre_restore_logical_to_physical: tuple[int, ...]
    final_logical_to_physical: tuple[int, ...]
    path_cache_delta: dict[str, int]
    direction_semantics: str = "logical_wire_order_preserved"
    direction_swap_count: int = 0
    expected_link_faults: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "source_instruction_count": self.source_instruction_count,
            "estimated_instruction_count": self.estimated_instruction_count,
            "topology_gate_count": self.topology_gate_count,
            "routed_gate_count": self.routed_gate_count,
            "planned_inserted_swap_count": self.planned_inserted_swap_count,
            "direction_semantics": self.direction_semantics,
            "direction_swap_count": self.direction_swap_count,
            "expected_link_faults": self.expected_link_faults,
            "skipped_channel_count": self.skipped_channel_count,
            "pre_restore_logical_to_physical": (self.pre_restore_logical_to_physical),
            "final_logical_to_physical": self.final_logical_to_physical,
            "path_cache_delta": dict(self.path_cache_delta),
        }


@dataclass(frozen=True)
class RoutingStrategySelection:
    """Auditable strategy choice made from the planning cost of every strategy.

    ``candidates`` holds one priced estimate per name in
    ``DEPLOYABLE_ROUTING_STRATEGIES``, in that order, so the record states both the
    choice and what it was chosen over. The set is the deployment contract's, not
    the whole routing vocabulary, because a choice the contract refuses is not a
    choice a caller can deploy. It is an immutable view rather than a dict a caller
    can edit, because the record is written into compile metadata as the evidence
    of the choice.

    ``objective`` names the ranking the choice actually used, which depends on
    whether the device was calibrated: a device that states only its couplings is
    ranked on counts, and one that states a per-link error rate is ranked on the
    fault total first. The string is derived from the selected estimate rather than
    stored, so the record cannot claim a ranking the estimate contradicts.
    """

    selected_strategy: str
    candidates: Mapping[str, RoutingCostEstimate]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))

    def summary(self) -> dict[str, Any]:
        chosen = self.candidates[self.selected_strategy]
        return {
            "schema": "flagquantum_routing_strategy_selection_v1",
            "objective": (
                "minimize_planned_inserted_swaps_then_instructions"
                if chosen.expected_link_faults is None
                else "minimize_expected_link_faults_then_planned_inserted_swaps_"
                "then_instructions"
            ),
            "selected_strategy": self.selected_strategy,
            "tie_breaker": "candidate_order",
            "candidates": {
                name: estimate.summary() for name, estimate in self.candidates.items()
            },
        }


def _estimate_swap_plan(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    *,
    strategy: str,
) -> tuple[int, tuple[int, ...], int, float | None]:
    """Estimate the SWAPs one estimating strategy inserts, and its mid-plan layout.

    The estimate replays the strategy's layout bookkeeping over the program's qubit
    pairs instead of emitting instructions, which is what keeps it cheap.
    ``restore_after_each_gate`` returns to the identity layout after every gate, so
    it pays each hop twice and ends where it started; ``persistent_layout`` keeps
    the layout it reached, so it pays each hop once and reports where it finished.

    The third result is the direction cost: the number of extra hops that exist
    only because the device orders its links, counted on the same placement the
    router emits -- the last hop of the path for ``restore_after_each_gate``, the
    post-move qubits for ``persistent_layout``. Those hops are already included in
    the first result, because an orientation SWAP is a SWAP like any other; the
    count is returned separately so the record states why the price is higher than
    the same program on the undirected projection of the same device.

    The fourth result is the same replay priced in faults rather than hops, and it
    is ``None`` unless the device states a per-link error rate. It is the summed
    two-qubit error probability of the operations the strategy would emit: every
    forward hop twice, because both strategies restore the layout by undoing their
    hops at the end of the program, the gate itself once on the link it runs on,
    and an orientation repair twice because it is a SWAP sandwich. That is the same
    accounting ``route_to_topology`` reports for the program it emits, arrived at
    without emitting it, and the two are held equal by test.

    Returns:
        ``(planned_inserted_swap_count, pre_restore_logical_to_physical,
        direction_swap_count, expected_link_faults)``.
    """

    logical_to_physical = list(range(ir.n_wires))
    physical_to_logical = list(range(ir.n_wires))

    def apply_swap(left: int, right: int) -> None:
        left_logical = physical_to_logical[left]
        right_logical = physical_to_logical[right]
        physical_to_logical[left], physical_to_logical[right] = (
            right_logical,
            left_logical,
        )
        logical_to_physical[left_logical] = right
        logical_to_physical[right_logical] = left

    calibrated = topology.link_error_rates is not None
    forward_swap_count = 0
    direction_swap_count = 0
    expected_link_faults = 0.0
    for instruction in ir:
        if not _is_routable_two_wire_gate(instruction):
            continue
        mapped_wires = tuple(logical_to_physical[wire] for wire in instruction.wires)
        if topology.executes(instruction, mapped_wires):
            placement = mapped_wires
            if calibrated:
                expected_link_faults += topology.link_error(*mapped_wires)
        else:
            path = topology.path(mapped_wires)
            if any(wire >= ir.n_wires for wire in path):
                raise ValueError(
                    "Routing through physical ancilla qubits outside the circuit IR "
                    "is not supported; provide an explicit layout/lowering step."
                )
            gate_swap_count = max(0, len(path) - 2)
            forward_swap_count += gate_swap_count
            if calibrated:
                for path_index in range(len(path) - 2):
                    expected_link_faults += 2 * topology.link_error(
                        path[path_index], path[path_index + 1]
                    )
                expected_link_faults += topology.link_error(path[-2], path[-1])
            if strategy == "persistent_layout":
                for path_index in range(gate_swap_count):
                    apply_swap(path[path_index], path[path_index + 1])
                placement = tuple(
                    logical_to_physical[wire] for wire in instruction.wires
                )
            else:
                placement = (path[-2], path[-1])
        if topology.needs_orientation_swap(instruction, placement):
            forward_swap_count += 1
            direction_swap_count += 1
            if calibrated:
                expected_link_faults += 2 * topology.link_error(*placement)
            if strategy == "persistent_layout":
                apply_swap(*placement)
    identity_layout = tuple(range(ir.n_wires))
    pre_restore_layout = (
        tuple(logical_to_physical)
        if strategy == "persistent_layout"
        else identity_layout
    )
    return (
        2 * forward_swap_count,
        pre_restore_layout,
        direction_swap_count,
        expected_link_faults if calibrated else None,
    )


def estimate_routing_cost(
    circuit_or_ir: Any,
    coupling_map: _PricedDevice,
    *,
    strategy: str = "restore_after_each_gate",
) -> RoutingCostEstimate:
    """Estimate one strategy's routing growth without emitting routed instructions.

    Every name in ``ROUTING_STRATEGIES`` is priceable. The two estimating
    strategies are priced by replaying their layout bookkeeping over the program's
    qubit pairs; the two SABRE strategies are priced by planning their SWAPs, which
    is the work ``route_to_topology`` would do anyway, so a planned price is the
    count the chosen strategy will insert rather than a bound on it. Pricing all
    four is what lets ``select_routing_strategy`` compare a planner against an
    estimate.

    A ``DirectedCouplingMap`` costs more than its own undirected projection, and
    the difference is the whole point: a gate placed on a link whose declared
    direction is the opposite of its operands needs one extra SWAP across that
    link, so an ordered device prices into ``planned_inserted_swap_count`` the
    hops that exist only because the links are ordered. ``direction_swap_count``
    states how many of those hops that is, so a caller can see the ordering cost
    rather than infer it from a total.

    A calibrated device is priced in faults as well as in SWAPs: the two
    shortest-path strategies report the summed link error of the operations they
    would emit, on the route the calibration chose. That total counts a SWAP as one
    operation and the gate as another, so it compares a short route over dirty links
    with a long route over clean ones -- which is the comparison a hop count cannot
    make. The planners behind ``sabre`` and
    ``sabre_layout`` are priced on the weak projection and repaired the same way,
    so every name in the vocabulary is comparable on one quantity.

    Args:
        circuit_or_ir: The program to price.
        coupling_map: The device to price it against, as a ``CouplingMap``, a
            ``DirectedCouplingMap``, a ``LinkErrorRates`` device, an edge sequence,
            or a cost model this module already built. A model is priced unchanged
            rather than rebuilt, which is what lets ``select_routing_strategy``
            price several strategies against one weak projection and one path
            cache.
        strategy: One of ``ROUTING_STRATEGIES``.

    Returns:
        The planned SWAP count, the expected instruction count, the source counts
        the same strategy would report in routing metadata, and -- on an ordered
        device -- how much of that SWAP count the ordering itself forced.

    Raises:
        TypeError: If ``coupling_map`` is not a device or a cost model.
        ValueError: If ``strategy`` is not in ``ROUTING_STRATEGIES``, the coupling
            map has fewer qubits than the circuit, a costed path leaves the
            device, a planner cannot route the program, a placed two-qubit opcode
            has no recorded operand semantics, or a calibrated device is priced
            with a strategy that does not read the calibration.
    """

    if strategy not in ROUTING_STRATEGIES:
        raise ValueError(
            "routing strategy must be one of "
            + ", ".join(repr(name) for name in ROUTING_STRATEGIES)
        )
    ir = ensure_circuit_ir(circuit_or_ir)
    topology = _topology_cost_model(coupling_map, ir.n_wires, "estimate_routing_cost")
    estimate, _ = _estimate_routing_cost(ir, topology, strategy=strategy)
    return estimate


def _plan_sabre_for_topology(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    initial_layout: tuple[int, ...],
) -> SabrePlan:
    coupling = topology.weak_coupling
    if not topology.is_ordered:
        return plan_sabre_swaps(ir, coupling, initial_layout=initial_layout)
    baseline = _plan_sabre_trial(ir, coupling, initial_layout=initial_layout)
    if baseline is None:
        raise RuntimeError("unbounded SABRE trial did not produce a plan")

    def cost(plan: SabrePlan) -> int:
        return (
            len(plan.swaps)
            + len(plan_restore_swaps(plan, coupling))
            + 2 * _count_sabre_orientation_swaps(ir, topology, plan.placements)
        )

    return _select_sabre_trials(ir, coupling, baseline, cost=cost)


def _estimate_routing_cost(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    *,
    strategy: str,
) -> tuple[RoutingCostEstimate, SabrePlan | None]:
    """Retain a planner's work beside its price for one compilation only."""
    if topology.n_qubits < ir.n_wires:
        raise ValueError("Coupling map has fewer qubits than the circuit.")

    identity_layout = tuple(range(ir.n_wires))
    cache_before = topology.path_cache_info()
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, topology)
    )
    plan: SabrePlan | None = None
    if strategy in SABRE_ROUTING_STRATEGIES:
        require_weight_reading_strategy(
            strategy,
            topology,
            "estimate_routing_cost",
            LINK_ERROR_ROUTING_STRATEGIES,
        )
        coupling = topology.weak_coupling
        initial_layout = (
            plan_sabre_layout(ir, coupling)
            if strategy == "sabre_layout"
            else identity_layout
        )
        plan = _plan_sabre_for_topology(ir, topology, initial_layout)
        direction_swap_count = _count_sabre_orientation_swaps(
            ir,
            topology,
            plan.placements,
        )
        # A repaired gate is wrapped in a SWAP sandwich rather than handed a new
        # layout, because the plan's remaining placements were costed against the
        # layout the planner chose; the sandwich returns to it.
        planned_swaps = (
            len(plan.swaps)
            + len(plan_restore_swaps(plan, coupling))
            + 2 * direction_swap_count
        )
        pre_restore_layout = plan.final_logical_to_physical
        expected_link_faults = None
    else:
        (
            planned_swaps,
            pre_restore_layout,
            direction_swap_count,
            expected_link_faults,
        ) = _estimate_swap_plan(ir, topology, strategy=strategy)
    cache_after = topology.path_cache_info()
    estimate = RoutingCostEstimate(
        strategy=strategy,
        source_instruction_count=len(ir),
        estimated_instruction_count=len(ir) + planned_swaps,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        planned_inserted_swap_count=planned_swaps,
        skipped_channel_count=skipped_channel_count,
        pre_restore_logical_to_physical=pre_restore_layout,
        final_logical_to_physical=identity_layout,
        path_cache_delta={
            key: cache_after[key] - cache_before[key]
            for key in ("hits", "misses", "evictions")
        },
        direction_semantics=topology.direction_semantics,
        direction_swap_count=direction_swap_count,
        expected_link_faults=expected_link_faults,
    )
    return estimate, plan


def select_routing_strategy(
    circuit_or_ir: Any,
    coupling_map: _PricedDevice,
) -> RoutingStrategySelection:
    """Select the cheapest routing strategy the deployment contract accepts.

    The candidates are ``DEPLOYABLE_ROUTING_STRATEGIES``, so the automatic choice
    can reach the ``sabre`` planner while never producing a program deployment
    refuses. Candidates are ranked by planned inserted SWAPs and then by expected
    instruction count; ties are broken by candidate order, which keeps the two
    estimating strategies ahead of the planner so that an equally priced plan is
    not planned twice.

    Pricing a planner is planning it. A separate call to ``route_to_topology``
    plans again; native ``compile`` retains the selected plan locally and
    materializes it without repeating the search.

    A calibrated device narrows the candidates to
    ``LINK_ERROR_ROUTING_STRATEGIES`` and ranks them on ``expected_link_faults``
    first, then on the two counts. The set is narrower because a strategy that
    chooses its routes by hop count cannot be ranked by a quantity it never read,
    and the ranking is fault-first because that is the question a caller who
    calibrated the device asked.

    A directed device ranks the candidates on the same two quantities, and both
    of them already include the ordering cost: a plan that has to repair a
    wrong-way gate pays for that repair in ``planned_inserted_swap_count`` and in
    ``estimated_instruction_count``, so the ranking compares the cost of routing
    on the ordered device rather than the cost on its undirected projection.
    ``direction_swap_count`` states how much of each candidate's price the ordering
    accounts for, so the choice comes with its reason rather than only its total.

    Raises:
        TypeError: If ``coupling_map`` is not a ``CouplingMap``, a
            ``DirectedCouplingMap``, a ``LinkErrorRates`` device, or an edge
            sequence.
        ValueError: If a placed two-qubit opcode has no recorded operand semantics,
            so an ordered device cannot price it.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    topology = _topology_cost_model(coupling_map, ir.n_wires, "select_routing_strategy")
    selection, _ = _select_routing_strategy(ir, topology)
    return selection


def _select_routing_strategy(
    ir: CircuitIR,
    topology: _TopologyCostModel,
) -> tuple[RoutingStrategySelection, SabrePlan | None]:
    calibrated = topology.link_error_rates is not None
    # A calibrated device narrows the candidates rather than widening the ranking
    # key, because a strategy that never reads the calibration cannot be ranked by
    # it: pricing `sabre` here would put a hop-count plan into a fault-ordered
    # comparison and call the result a choice.
    considered = (
        LINK_ERROR_ROUTING_STRATEGIES if calibrated else DEPLOYABLE_ROUTING_STRATEGIES
    )
    estimates_and_plans = {
        name: _estimate_routing_cost(ir, topology, strategy=name) for name in considered
    }
    candidates = {name: item[0] for name, item in estimates_and_plans.items()}
    selected = min(
        considered,
        key=lambda name: (
            (
                candidates[name].expected_link_faults
                if candidates[name].expected_link_faults is not None
                else 0.0
            ),
            candidates[name].planned_inserted_swap_count,
            candidates[name].estimated_instruction_count,
        ),
    )
    return (
        RoutingStrategySelection(selected_strategy=selected, candidates=candidates),
        estimates_and_plans[selected][1],
    )
