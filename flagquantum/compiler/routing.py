"""Topology-aware SWAP routing over an undirected or ordered coupling map.

The two device types this module routes against are declared next door:
``flagquantum/compiler/coupling.py`` owns the undirected ``CouplingMap`` and
``flagquantum/compiler/directed_topology.py`` owns the ordered one. Both are
re-exported here, because the routing entries are what a consumer asks first and
a caller holding a routing plan should not have to learn which module defines the
device the plan names.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from .coupling import UNREACHABLE_DISTANCE as UNREACHABLE_DISTANCE
from .coupling import CouplingMap as CouplingMap
from .directed_topology import DirectedCouplingMap
from .operand_semantics import (
    _operand_symmetry,
    _require_multi_wire_device_local,
)
from .sabre import plan_restore_swaps, plan_sabre_layout, plan_sabre_swaps

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


@dataclass(frozen=True)
class RoutingCostEstimate:
    """Lightweight routing size estimate without materializing instructions.

    ``direction_swap_count`` is the part of ``planned_inserted_swap_count`` an
    undirected device cannot produce: it counts the hops inserted only because a
    gate's operands run against the direction its link is declared in. It is zero
    on an undirected device, where no link has a direction to run against, and
    ``direction_semantics`` states which of the two devices the estimate was taken
    against.
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
    """

    selected_strategy: str
    candidates: Mapping[str, RoutingCostEstimate]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_routing_strategy_selection_v1",
            "objective": "minimize_planned_inserted_swaps_then_instructions",
            "selected_strategy": self.selected_strategy,
            "tie_breaker": "candidate_order",
            "candidates": {
                name: estimate.summary() for name, estimate in self.candidates.items()
            },
        }


def _is_routable_two_wire_gate(instruction: Instruction) -> bool:
    """Report whether a routing strategy moves this instruction's wires.

    No strategy moves a channel, and an instruction with fewer than two wires has
    no coupling to satisfy, so both stay where they are written and neither is
    part of a strategy's cost. The two readings that place a gate -- the SWAP
    estimate and a planner's placements -- have to answer this question the same
    way, or the price and the plan disagree about which gates the strategy moved.
    """

    return len(instruction.wires) == 2 and not instruction.metadata.get("is_channel")


class _TopologyCostModel:
    """The device facts a routing cost model reads, ordered or unordered.

    A cost model asks its device four things: how many physical wires it has,
    which couplings it declares, whether a two-wire instruction can execute where
    it stands, and -- when it cannot -- which physical path to move it along. On an
    undirected ``CouplingMap`` the third question is the second one, because
    either direction of a link satisfies every two-wire instruction. On a
    ``DirectedCouplingMap`` they differ: an operand-symmetric opcode such as
    ``swap`` may use either direction of a link, while a control/target opcode
    such as ``cx`` may use only the direction its operands are written in. That
    rule is read from
    ``flagquantum.compiler.operand_semantics._operand_symmetry``, which is also
    what direction legalization reads, so the two cannot drift apart.

    A wrong-way placement is repaired with the router's own vocabulary: one SWAP
    across the offending link exchanges the two operands, so the gate executes in
    the direction the device declares. The SWAP is symmetric, so it needs only the
    weak link, and it is layout-neutral when the surrounding strategy restores the
    mapping afterwards. That is why direction belongs to a routing cost model at
    all: on an ordered device it is priced in SWAPs, not merely reported.

    Two questions are deliberately not asked of the ordered edges. Paths and the
    SABRE planners search the weak projection, because inserting a SWAP across a
    link is legal in either direction; and multi-wire locality is checked on weak
    edges, because a three-wire instruction has no operand order to satisfy on the
    couplings it spans. Direction decides whether a placed gate is executable, not
    whether the router may cross the link.
    """

    __slots__ = ("_device", "_weak")

    def __init__(self, device: CouplingMap | DirectedCouplingMap) -> None:
        self._device = device
        self._weak = (
            device
            if isinstance(device, CouplingMap)
            else CouplingMap(device.n_wires, device.edges)
        )

    @property
    def n_wires(self) -> int:
        return self._device.n_wires

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return the couplings as the device declares them, direction included."""

        return self._device.edges

    @property
    def is_ordered(self) -> bool:
        return isinstance(self._device, DirectedCouplingMap)

    @property
    def direction_semantics(self) -> str:
        return "directed_cx" if self.is_ordered else "logical_wire_order_preserved"

    @property
    def weak_coupling(self) -> CouplingMap:
        """Return the undirected graph the planners search."""

        return self._weak

    def path_cache_info(self) -> dict[str, int]:
        return self._weak.path_cache_info()

    def weak_edge(self, left: int, right: int) -> bool:
        return self._weak.has_edge(left, right)

    def path(self, wires: tuple[int, ...]) -> tuple[int, ...]:
        """Return the wire sequence a SWAP series must follow to join two wires."""

        return self._weak.shortest_path(*wires)

    def executes(self, instruction: Instruction, wires: tuple[int, ...]) -> bool:
        """Report whether the device runs this instruction on these wires.

        Raises:
            ValueError: A directed device is asked about a two-wire opcode with no
                recorded operand semantics, so no direction rule applies and any
                answer would be a guess. Routing such a device is not a fallback
                decision: the opcode has to be classified before either the device
                or the program can be priced.
        """

        if not self.is_ordered:
            return self._weak.has_edge(*wires)
        control, target = wires
        symmetric = _operand_symmetry(instruction)
        if symmetric is None:
            raise ValueError(
                f"two-wire instruction {instruction.name!r} has no recorded operand "
                "semantics, so an ordered device cannot price it: record the opcode "
                "in flagquantum.core.operator_schema.OPERATOR_SCHEMAS"
            )
        if symmetric:
            return self._device.has_weak_edge(control, target)
        return self._device.has_edge(control, target)

    def needs_orientation_swap(
        self,
        instruction: Instruction,
        wires: tuple[int, ...],
    ) -> bool:
        """Report whether a placed gate is on its link the wrong way round.

        True exactly when the device declares the link but not the direction these
        operands are written in, so one SWAP across the link makes the gate
        executable. An opcode with no recorded operand semantics, and a pair with
        no link to run along at all, are not this case: the first is refused
        earlier by ``executes``, and the second is unrouted rather than reversed.
        """

        if not self.is_ordered:
            return False
        if _operand_symmetry(instruction) is not False:
            return False
        return self._weak.has_edge(*wires) and not self._device.has_edge(*wires)


def _topology_cost_model(
    coupling_map: object,
    wire_count: int,
    entry: str,
) -> _TopologyCostModel:
    """Coerce the device a cost-model entry was handed.

    A directed device is a device the cost model reads, not one it refuses: the
    ordered edges decide which placed gates need a reversal, and the weak
    projection decides where the SWAPs go.

    A model is returned unchanged so that a caller pricing several strategies --
    ``select_routing_strategy`` prices three -- builds one weak projection and one
    path cache, and so measures a later candidate's cache traffic against the
    earlier candidates' work rather than against a cold device.
    """

    if isinstance(coupling_map, _TopologyCostModel):
        return coupling_map
    if isinstance(coupling_map, (CouplingMap, DirectedCouplingMap)):
        return _TopologyCostModel(coupling_map)
    if not isinstance(coupling_map, Iterable):
        raise TypeError(
            f"{entry} takes a CouplingMap, a DirectedCouplingMap, or an iterable "
            f"of (control, target) edges, not a {type(coupling_map).__name__}"
        )
    return _TopologyCostModel(CouplingMap(wire_count, coupling_map))


def _source_routing_counts(
    ir: CircuitIR,
    topology: _TopologyCostModel,
) -> tuple[int, int, int]:
    """Return the source program's topology, routed, and skipped-channel counts.

    These three describe a program against one device, and every strategy reports
    them the same way: a two-wire operation is a topology gate, one the device
    cannot execute as written is also a routed gate, and a two-wire channel is
    neither because routing may not move a channel. On an ordered device "as
    written" is a claim about the operands, so a CX whose operands run against
    the direction its link is declared in counts as routed work here rather than
    as a gate the device already carries, while the same CX the other way round
    is executed where it stands and is not counted. Counting them in one place is
    what keeps a strategy's routing metadata and its cost estimate from
    disagreeing about the same program.
    """

    topology_gate_count = 0
    routed_gate_count = 0
    skipped_channel_count = 0
    for instruction in ir:
        if len(instruction.wires) != 2:
            continue
        if instruction.metadata.get("is_channel"):
            skipped_channel_count += 1
            continue
        topology_gate_count += 1
        if not topology.executes(instruction, instruction.wires):
            routed_gate_count += 1
    return topology_gate_count, routed_gate_count, skipped_channel_count


def _estimate_swap_plan(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    *,
    strategy: str,
) -> tuple[int, tuple[int, ...], int]:
    """Estimate the SWAPs one estimating strategy inserts, and its mid-plan layout.

    The estimate replays the strategy's layout bookkeeping over the program's wire
    pairs instead of emitting instructions, which is what keeps it cheap.
    ``restore_after_each_gate`` returns to the identity layout after every gate, so
    it pays each hop twice and ends where it started; ``persistent_layout`` keeps
    the layout it reached, so it pays each hop once and reports where it finished.

    The third result is the direction cost: the number of extra hops that exist
    only because the device orders its links, counted on the same placement the
    router emits -- the last hop of the path for ``restore_after_each_gate``, the
    post-move wires for ``persistent_layout``. Those hops are already included in
    the first result, because an orientation SWAP is a SWAP like any other; the
    count is returned separately so the record states why the price is higher than
    the same program on the undirected projection of the same device.

    Returns:
        ``(planned_inserted_swap_count, pre_restore_logical_to_physical,
        direction_swap_count)``.
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

    forward_swap_count = 0
    direction_swap_count = 0
    for instruction in ir:
        if not _is_routable_two_wire_gate(instruction):
            continue
        mapped_wires = tuple(logical_to_physical[wire] for wire in instruction.wires)
        if topology.executes(instruction, mapped_wires):
            placement = mapped_wires
        else:
            path = topology.path(mapped_wires)
            if any(wire >= ir.n_wires for wire in path):
                raise ValueError(
                    "Routing through physical ancilla wires outside the circuit IR "
                    "is not supported; provide an explicit layout/lowering step."
                )
            gate_swap_count = max(0, len(path) - 2)
            forward_swap_count += gate_swap_count
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
            if strategy == "persistent_layout":
                apply_swap(*placement)
    identity_layout = tuple(range(ir.n_wires))
    pre_restore_layout = (
        tuple(logical_to_physical)
        if strategy == "persistent_layout"
        else identity_layout
    )
    return 2 * forward_swap_count, pre_restore_layout, direction_swap_count


def _count_sabre_orientation_swaps(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    placements: tuple[tuple[int, ...], ...],
) -> int:
    """Count the gates one planned placement leaves on their link the wrong way.

    A planner reports where it places every source instruction, so the count is
    read off that placement rather than re-derived, and the price of a planned plan
    describes the plan the caller will get. Each counted gate costs two SWAPs: the
    materializer wraps it in a SWAP sandwich, so the plan's own layout bookkeeping
    survives the repair and the rest of the plan stays valid.
    """

    return sum(
        1
        for index, instruction in enumerate(ir)
        if _is_routable_two_wire_gate(instruction)
        and topology.needs_orientation_swap(instruction, tuple(placements[index]))
    )


def estimate_routing_cost(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | DirectedCouplingMap | Iterable[tuple[int, int]],
    *,
    strategy: str = "restore_after_each_gate",
) -> RoutingCostEstimate:
    """Estimate one strategy's routing growth without emitting routed instructions.

    Every name in ``ROUTING_STRATEGIES`` is priceable. The two estimating
    strategies are priced by replaying their layout bookkeeping over the program's
    wire pairs; the two SABRE strategies are priced by planning their SWAPs, which
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
    rather than infer it from a total. The planners behind ``sabre`` and
    ``sabre_layout`` are priced on the weak projection and repaired the same way,
    so every name in the vocabulary is comparable on one quantity.

    Args:
        circuit_or_ir: The program to price.
        coupling_map: The device to price it against, as a ``CouplingMap``, a
            ``DirectedCouplingMap``, or an edge sequence.
        strategy: One of ``ROUTING_STRATEGIES``.

    Returns:
        The planned SWAP count, the expected instruction count, the source counts
        the same strategy would report in routing metadata, and -- on an ordered
        device -- how much of that SWAP count the ordering itself forced.

    Raises:
        TypeError: If ``coupling_map`` is none of the three accepted forms.
        ValueError: If ``strategy`` is not in ``ROUTING_STRATEGIES``, the coupling
            map has fewer wires than the circuit, a costed path leaves the
            device, a planner cannot route the program, or a placed two-wire
            opcode has no recorded operand semantics.
    """

    if strategy not in ROUTING_STRATEGIES:
        raise ValueError(
            "routing strategy must be one of "
            + ", ".join(repr(name) for name in ROUTING_STRATEGIES)
        )
    ir = ensure_circuit_ir(circuit_or_ir)
    topology = _topology_cost_model(coupling_map, ir.n_wires, "estimate_routing_cost")
    if topology.n_wires < ir.n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")

    identity_layout = tuple(range(ir.n_wires))
    cache_before = topology.path_cache_info()
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, topology)
    )
    if strategy in SABRE_ROUTING_STRATEGIES:
        coupling = topology.weak_coupling
        initial_layout = (
            plan_sabre_layout(ir, coupling)
            if strategy == "sabre_layout"
            else identity_layout
        )
        plan = plan_sabre_swaps(ir, coupling, initial_layout=initial_layout)
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
    else:
        planned_swaps, pre_restore_layout, direction_swap_count = _estimate_swap_plan(
            ir, topology, strategy=strategy
        )
    cache_after = topology.path_cache_info()
    return RoutingCostEstimate(
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
    )


def select_routing_strategy(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | DirectedCouplingMap | Iterable[tuple[int, int]],
) -> RoutingStrategySelection:
    """Select the cheapest routing strategy the deployment contract accepts.

    The candidates are ``DEPLOYABLE_ROUTING_STRATEGIES``, so the automatic choice
    can reach the ``sabre`` planner while never producing a program deployment
    refuses. Candidates are ranked by planned inserted SWAPs and then by expected
    instruction count; ties are broken by candidate order, which keeps the two
    estimating strategies ahead of the planner so that an equally priced plan is
    not planned twice.

    Pricing a planner is planning it, so this call does the work
    ``route_to_topology`` will repeat. A caller who routes once should pass the
    strategy it wants rather than asking for ``"auto"``.

    A directed device ranks the candidates on the same two quantities, and both
    of them already include the ordering cost: a plan that has to repair a
    wrong-way gate pays for that repair in ``planned_inserted_swap_count`` and in
    ``estimated_instruction_count``, so the ranking compares the cost of routing
    on the ordered device rather than the cost on its undirected projection.
    ``direction_swap_count`` states how much of each candidate's price the ordering
    accounts for, so the choice comes with its reason rather than only its total.

    Raises:
        TypeError: If ``coupling_map`` is not a ``CouplingMap``, a
            ``DirectedCouplingMap``, or an edge sequence.
        ValueError: If a placed two-wire opcode has no recorded operand semantics,
            so an ordered device cannot price it.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    topology = _topology_cost_model(coupling_map, ir.n_wires, "select_routing_strategy")
    candidates = {
        name: estimate_routing_cost(ir, topology, strategy=name)
        for name in DEPLOYABLE_ROUTING_STRATEGIES
    }
    selected = min(
        DEPLOYABLE_ROUTING_STRATEGIES,
        key=lambda name: (
            candidates[name].planned_inserted_swap_count,
            candidates[name].estimated_instruction_count,
        ),
    )
    return RoutingStrategySelection(selected_strategy=selected, candidates=candidates)


def _swap_instruction(
    left: int,
    right: int,
    source: Instruction,
    *,
    strategy: str,
    phase: str,
    source_instruction_index: int,
) -> Instruction:
    return Instruction(
        name="swap",
        wires=(left, right),
        metadata={
            "routed_from": source.name,
            "routing_strategy": strategy,
            "routing_phase": phase,
            "logical_wires": source.wires,
            "source_instruction_index": source_instruction_index,
        },
    )


def _routing_metadata(
    *,
    ir: CircuitIR,
    topology: _TopologyCostModel,
    strategy: str,
    topology_gate_count: int,
    routed_gate_count: int,
    inserted_swap_count: int,
    direction_swap_count: int,
    skipped_channel_count: int,
    initial_layout: tuple[int, ...],
    pre_restore_layout: tuple[int, ...],
    path_cache_before: dict[str, int],
    path_cache_after: dict[str, int],
) -> dict[str, Any]:
    identity_layout = tuple(range(ir.n_wires))
    return {
        "schema": "flagquantum_routing_plan_v1",
        "strategy": strategy,
        "coupling_n_wires": topology.n_wires,
        "coupling_edges": topology.edges,
        "initial_logical_to_physical": initial_layout,
        "pre_restore_logical_to_physical": pre_restore_layout,
        "final_logical_to_physical": identity_layout,
        "mapping_restored": True,
        "direction_semantics": topology.direction_semantics,
        "direction_swap_count": direction_swap_count,
        "topology_gate_count": topology_gate_count,
        "routed_gate_count": routed_gate_count,
        "inserted_swap_count": inserted_swap_count,
        "planned_inserted_swap_count": inserted_swap_count,
        "post_optimization_inserted_swap_count": None,
        "skipped_channel_count": skipped_channel_count,
        "persistent_layout_supported": True,
        "path_cache": {
            "before": path_cache_before,
            "after": path_cache_after,
            "delta": {
                key: path_cache_after[key] - path_cache_before[key]
                for key in ("hits", "misses", "evictions")
            },
        },
    }


def _remap_instruction(
    instruction: Instruction,
    wires: tuple[int, ...],
    *,
    strategy: str,
    source_instruction_index: int,
) -> Instruction:
    return Instruction(
        name=instruction.name,
        wires=wires,
        params=instruction.params,
        matrix=instruction.matrix,
        metadata=dict(instruction.metadata)
        | {
            "layout_mapped": wires != instruction.wires,
            "logical_wires": instruction.wires,
            "routing_strategy": strategy,
            "source_instruction_index": source_instruction_index,
        },
    )


def _route_persistent_layout(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    *,
    path_cache_before: dict[str, int],
) -> CircuitIR:
    strategy = "persistent_layout"
    logical_to_physical = list(range(ir.n_wires))
    physical_to_logical = list(range(ir.n_wires))
    routed: list[Instruction] = []
    routing_swaps: list[tuple[int, int, Instruction, int]] = []
    direction_swap_count = 0
    # Counted against the source wires, like every other strategy and like the
    # cost estimate: a gate this layout happened to make local is still a gate
    # the device did not carry, and reporting it as unrouted would make the same
    # field mean something different per strategy.
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, topology)
    )

    def apply_mapping_swap(
        left: int,
        right: int,
        source: Instruction,
        *,
        phase: str,
        source_instruction_index: int,
    ) -> None:
        routed.append(
            _swap_instruction(
                left,
                right,
                source,
                strategy=strategy,
                phase=phase,
                source_instruction_index=source_instruction_index,
            )
        )
        left_logical = physical_to_logical[left]
        right_logical = physical_to_logical[right]
        physical_to_logical[left], physical_to_logical[right] = (
            right_logical,
            left_logical,
        )
        logical_to_physical[left_logical] = right
        logical_to_physical[right_logical] = left

    for source_index, instruction in enumerate(ir):
        mapped_wires = tuple(logical_to_physical[wire] for wire in instruction.wires)
        if len(instruction.wires) != 2:
            _require_multi_wire_device_local(
                instruction, mapped_wires, topology.weak_edge
            )
            routed.append(
                _remap_instruction(
                    instruction,
                    mapped_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        if instruction.metadata.get("is_channel"):
            routed.append(
                _remap_instruction(
                    instruction,
                    mapped_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        left, right = mapped_wires
        if not topology.executes(instruction, mapped_wires):
            path = topology.path(mapped_wires)
            if any(wire >= ir.n_wires for wire in path):
                raise ValueError(
                    "Routing through physical ancilla wires outside the circuit IR "
                    "is not supported; provide an explicit layout/lowering step."
                )
            for path_index in range(len(path) - 2):
                swap = (
                    path[path_index],
                    path[path_index + 1],
                    instruction,
                    source_index,
                )
                apply_mapping_swap(
                    swap[0],
                    swap[1],
                    swap[2],
                    phase="forward",
                    source_instruction_index=source_index,
                )
                routing_swaps.append(swap)
            mapped_wires = tuple(
                logical_to_physical[wire] for wire in instruction.wires
            )
        # A link whose declared direction is the opposite of these operands needs
        # one more hop. It is kept in the mapping like every other hop this
        # strategy takes, so the same final restore returns it and the estimate's
        # doubled hop count describes the program that is emitted.
        if topology.needs_orientation_swap(instruction, mapped_wires):
            apply_mapping_swap(
                mapped_wires[0],
                mapped_wires[1],
                instruction,
                phase="orientation",
                source_instruction_index=source_index,
            )
            routing_swaps.append(
                (mapped_wires[0], mapped_wires[1], instruction, source_index)
            )
            mapped_wires = tuple(
                logical_to_physical[wire] for wire in instruction.wires
            )
            direction_swap_count += 1
        routed.append(
            _remap_instruction(
                instruction,
                mapped_wires,
                strategy=strategy,
                source_instruction_index=source_index,
            )
        )

    pre_restore_layout = tuple(logical_to_physical)
    for left, right, source, source_index in reversed(routing_swaps):
        apply_mapping_swap(
            left,
            right,
            source,
            phase="final_restore",
            source_instruction_index=source_index,
        )
    if logical_to_physical != list(range(ir.n_wires)):
        raise RuntimeError("persistent routing failed to restore the final layout")

    metadata = dict(ir.metadata)
    metadata["routing"] = _routing_metadata(
        ir=ir,
        topology=topology,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=2 * len(routing_swaps),
        direction_swap_count=direction_swap_count,
        skipped_channel_count=skipped_channel_count,
        initial_layout=tuple(range(ir.n_wires)),
        pre_restore_layout=pre_restore_layout,
        path_cache_before=path_cache_before,
        path_cache_after=topology.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def _route_sabre(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    *,
    strategy: str,
    path_cache_before: dict[str, int],
) -> CircuitIR:
    """Materialize one SABRE persistent layout, then restore the output layout.

    Instructions are emitted in the order the planner placed them, not in source
    order. The planner introduces a SWAP when a two-wire operation becomes
    reachable, and every operation keeps its planned physical wires, so a source
    order that disagrees with the plan would replay a different layout than the
    one the plan was costed against.

    The ``sabre`` strategy starts from the identity layout, so replaying its
    forward SWAPs in reverse restores the output layout. ``sabre_layout`` first
    searches for a shorter initial layout; replaying its forward SWAPs in reverse
    would only return that initial layout, so it routes every logical wire home
    with an explicit restore instead.

    On an ordered device a placed gate whose operands run against its link is
    wrapped in a SWAP sandwich. The sandwich is layout-neutral, which is what the
    planner's own placements require: they were costed against one layout, so a
    repair that moved a wire would invalidate every placement after it. The
    planner itself is handed the weak projection, because a SWAP may cross a link
    in either direction.
    """

    coupling = topology.weak_coupling
    initial_layout = (
        plan_sabre_layout(ir, coupling)
        if strategy == "sabre_layout"
        else tuple(range(ir.n_wires))
    )
    plan = plan_sabre_swaps(ir, coupling, initial_layout=initial_layout)
    routed: list[Instruction] = []
    # Counted against the source wires rather than the placed ones, exactly as the
    # shortest-path strategies and the cost estimate count them, so the number
    # describes the work the topology forces rather than the work this layout
    # happened to arrange for free.
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, topology)
    )
    direction_swap_count = 0
    swap_index = 0
    placed_so_far = 0
    for source_index in plan.sequence:
        while (
            swap_index < len(plan.swaps)
            and plan.swap_placements[swap_index] == placed_so_far
        ):
            anchor_index = plan.swap_anchors[swap_index]
            swap_left, swap_right = plan.swaps[swap_index]
            routed.append(
                _swap_instruction(
                    swap_left,
                    swap_right,
                    ir.instructions[anchor_index],
                    strategy=strategy,
                    phase="forward",
                    source_instruction_index=anchor_index,
                )
            )
            swap_index += 1
        placed_so_far += 1
        instruction = ir.instructions[source_index]
        placed_wires = plan.placements[source_index]
        if len(instruction.wires) != 2 or instruction.metadata.get("is_channel"):
            _require_multi_wire_device_local(
                instruction, placed_wires, topology.weak_edge
            )
            routed.append(
                _remap_instruction(
                    instruction,
                    placed_wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        gate_wires = placed_wires
        orientation = topology.needs_orientation_swap(instruction, placed_wires)
        if orientation:
            routed.append(
                _swap_instruction(
                    placed_wires[0],
                    placed_wires[1],
                    instruction,
                    strategy=strategy,
                    phase="orientation",
                    source_instruction_index=source_index,
                )
            )
            # The SWAP exchanges the operands' states, so the gate executes
            # between the two halves of the sandwich with its operands swapped;
            # the closing SWAP puts both wires back before the next planned
            # instruction, which is what keeps the planner's remaining placements
            # valid. Emitting both halves together would apply the gate on the
            # restored wires and silently undo the repair.
            gate_wires = (placed_wires[1], placed_wires[0])
            direction_swap_count += 1
        routed.append(
            _remap_instruction(
                instruction,
                gate_wires,
                strategy=strategy,
                source_instruction_index=source_index,
            )
        )
        if orientation:
            routed.append(
                _swap_instruction(
                    placed_wires[0],
                    placed_wires[1],
                    instruction,
                    strategy=strategy,
                    phase="orientation_restore",
                    source_instruction_index=source_index,
                )
            )
    if swap_index != len(plan.swaps):
        raise RuntimeError("SABRE routing plan did not place every inserted SWAP")
    restore = plan_restore_swaps(plan, coupling)
    for swap_left, swap_right, anchor_index in restore:
        routed.append(
            _swap_instruction(
                swap_left,
                swap_right,
                ir.instructions[anchor_index],
                strategy=strategy,
                phase="final_restore",
                source_instruction_index=anchor_index,
            )
        )
    metadata = dict(ir.metadata)
    metadata["routing"] = _routing_metadata(
        ir=ir,
        topology=topology,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=len(plan.swaps) + len(restore) + 2 * direction_swap_count,
        direction_swap_count=direction_swap_count,
        skipped_channel_count=skipped_channel_count,
        initial_layout=initial_layout,
        pre_restore_layout=plan.final_logical_to_physical,
        path_cache_before=path_cache_before,
        path_cache_after=topology.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def route_to_topology(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | DirectedCouplingMap | Iterable[tuple[int, int]],
    *,
    strategy: str = "restore_after_each_gate",
) -> CircuitIR:
    """Insert SWAP gates so two-qubit operations respect hardware topology.

    An undirected device is routed as before: a gate is executable wherever its
    two wires are linked, so a hop is inserted only to bring the wires together.

    An ordered device is routed against the direction of its links as well. A gate
    whose operands run against the direction its link is declared in is not
    executable where it stands even though the link exists, and the router repairs
    it with the one instruction that is legal across that link either way: a SWAP
    between the operands. That repair costs a SWAP and leaves the output layout
    where the strategy intended, because every strategy here either restores the
    identity layout or wraps the gate -- the persistent strategies keep the repair
    in their mapping and undo it with their own final restore, while
    ``restore_after_each_gate`` sandwiches the gate. Emitting the repair is what
    keeps ``direction_swap_count`` a price of this router rather than a bill handed
    to a later pass.

    Args:
        circuit_or_ir: The program to route.
        coupling_map: The device to route against, as a ``CouplingMap``, a
            ``DirectedCouplingMap``, or an edge sequence.
        strategy: One of ``ROUTING_STRATEGIES``.

    Raises:
        TypeError: If ``coupling_map`` is none of the three accepted forms.
        ValueError: If ``strategy`` is not in ``ROUTING_STRATEGIES``, the coupling
            map has fewer wires than the circuit, a needed path leaves the device,
            or a placed two-wire opcode has no recorded operand semantics on an
            ordered device.
    """

    if strategy not in ROUTING_STRATEGIES:
        raise ValueError(
            "routing strategy must be one of "
            + ", ".join(repr(name) for name in ROUTING_STRATEGIES)
        )
    ir = ensure_circuit_ir(circuit_or_ir)
    topology = _topology_cost_model(coupling_map, ir.n_wires, "route_to_topology")
    if topology.n_wires < ir.n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")
    path_cache_before = topology.path_cache_info()
    if strategy == "persistent_layout":
        return _route_persistent_layout(
            ir,
            topology,
            path_cache_before=path_cache_before,
        )
    if strategy in SABRE_ROUTING_STRATEGIES:
        return _route_sabre(
            ir,
            topology,
            strategy=strategy,
            path_cache_before=path_cache_before,
        )

    routed: list[Instruction] = []
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, topology)
    )
    inserted_swap_count = 0
    direction_swap_count = 0
    for source_index, instruction in enumerate(ir):
        if len(instruction.wires) != 2:
            _require_multi_wire_device_local(
                instruction, instruction.wires, topology.weak_edge
            )
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue
        if instruction.metadata.get("is_channel"):
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        if topology.executes(instruction, instruction.wires):
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        path = topology.path(instruction.wires)
        if any(wire >= ir.n_wires for wire in path):
            raise ValueError(
                "Routing through physical ancilla wires outside the circuit IR "
                "is not supported; provide an explicit layout/lowering step."
            )
        forward_swaps = [
            (path[index], path[index + 1]) for index in range(len(path) - 2)
        ]
        placement = (path[-2], path[-1])
        orientation = topology.needs_orientation_swap(instruction, placement)
        inserted_swap_count += 2 * (len(forward_swaps) + int(orientation))
        direction_swap_count += int(orientation)
        for swap_left, swap_right in forward_swaps:
            routed.append(
                _swap_instruction(
                    swap_left,
                    swap_right,
                    instruction,
                    strategy=strategy,
                    phase="forward",
                    source_instruction_index=source_index,
                )
            )
        if orientation:
            routed.append(
                _swap_instruction(
                    placement[0],
                    placement[1],
                    instruction,
                    strategy=strategy,
                    phase="orientation",
                    source_instruction_index=source_index,
                )
            )
            # The SWAP exchanges the two operands' states, so the gate now runs
            # with its control on ``placement[1]`` and its target on
            # ``placement[0]`` -- which is the direction the device declares,
            # since that is exactly what asked for the SWAP.
            placement = (placement[1], placement[0])
        routed.append(
            Instruction(
                name=instruction.name,
                wires=placement,
                params=instruction.params,
                matrix=instruction.matrix,
                metadata=dict(instruction.metadata)
                | {
                    "routed": True,
                    "logical_wires": instruction.wires,
                    "routing_strategy": strategy,
                    "source_instruction_index": source_index,
                },
            )
        )
        if orientation:
            routed.append(
                _swap_instruction(
                    placement[0],
                    placement[1],
                    instruction,
                    strategy=strategy,
                    phase="orientation_restore",
                    source_instruction_index=source_index,
                )
            )
        for swap_left, swap_right in reversed(forward_swaps):
            routed.append(
                _swap_instruction(
                    swap_left,
                    swap_right,
                    instruction,
                    strategy=strategy,
                    phase="gate_restore",
                    source_instruction_index=source_index,
                )
            )
    metadata = dict(ir.metadata)
    metadata["routing"] = _routing_metadata(
        ir=ir,
        topology=topology,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=inserted_swap_count,
        direction_swap_count=direction_swap_count,
        skipped_channel_count=skipped_channel_count,
        initial_layout=tuple(range(ir.n_wires)),
        pre_restore_layout=tuple(range(ir.n_wires)),
        path_cache_before=path_cache_before,
        path_cache_after=topology.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def record_post_routing_optimization(ir: CircuitIR) -> CircuitIR:
    """Attach retained routing-SWAP counts after compiler optimization."""

    routing = ir.metadata.get("routing")
    if not isinstance(routing, dict):
        return ir
    # ``routing_phase`` is the marker every strategy stamps on an inserted SWAP,
    # so the count does not depend on a list of strategy names.
    retained_swap_count = sum(
        instruction.name == "swap"
        and instruction.metadata.get("routing_phase") is not None
        for instruction in ir
    )
    metadata = dict(ir.metadata)
    metadata["routing"] = dict(routing) | {
        "post_optimization_inserted_swap_count": retained_swap_count,
        "post_optimization_instruction_count": len(ir),
    }
    return replace(ir, metadata=metadata)


__all__ = [
    "CouplingMap",
    "RoutingCostEstimate",
    "RoutingStrategySelection",
    "estimate_routing_cost",
    "record_post_routing_optimization",
    "route_to_topology",
    "select_routing_strategy",
]
