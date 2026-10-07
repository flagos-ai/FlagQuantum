"""Compiler-owned coupling maps and topology-aware SWAP routing."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from threading import Lock
from types import MappingProxyType
from typing import Any

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from .sabre import plan_restore_swaps, plan_sabre_layout, plan_sabre_swaps

# Dense distance-matrix entry for a physical wire pair with no coupling path.
# A coupling map is a graph, so most pairs of a production device are far
# apart but still reachable; this value is reserved for genuine disconnection.
UNREACHABLE_DISTANCE = -1

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

# The physical couplings a multi-wire instruction needs, as operand index pairs.
# This is the authoritative vocabulary for multi-wire locality in the Compiler:
# the pairs are the instruction's real two-wire interactions, taken from Core's
# operand order (`ccx` is control1/control2/target and conjugates on the target;
# `cswap` is control/target1/target2 and exchanges under the control), so the
# requirement is not that the operands form a chain. The opcodes are Core's
# arity-three operands, and
# `tests/team/compiler/test_multi_wire_routing_locality.py` fails when a Core
# arity of three or more has no entry here.
_MULTI_WIRE_OPERAND_PAIRS: dict[str, tuple[tuple[int, int], ...]] = {
    "ccx": ((0, 2), (1, 2)),
    "cswap": ((0, 1), (0, 2)),
}


def _breadth_first_distances(
    adjacency: tuple[tuple[int, ...], ...],
    source: int,
) -> tuple[int, ...]:
    """Return undirected hop counts from one wire to every wire."""

    distances = [UNREACHABLE_DISTANCE] * len(adjacency)
    distances[source] = 0
    queue: deque[int] = deque((source,))
    while queue:
        wire = queue.popleft()
        level = distances[wire] + 1
        for neighbor in adjacency[wire]:
            if distances[neighbor] == UNREACHABLE_DISTANCE:
                distances[neighbor] = level
                queue.append(neighbor)
    return tuple(distances)


@dataclass(frozen=True)
class CouplingMap:
    """Undirected hardware connectivity used by native routing passes."""

    n_wires: int
    edges: tuple[tuple[int, int], ...]
    _adjacency: tuple[tuple[int, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _path_cache: OrderedDict[tuple[int, int], tuple[int, ...]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    path_cache_capacity: int = field(default=4096, compare=False, hash=False)
    _path_cache_lock: Lock = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _path_cache_hits: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _path_cache_misses: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _path_cache_evictions: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _distance_rows: OrderedDict[int, tuple[int, ...]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _distance_hits: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _distance_misses: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _distance_evictions: int = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __init__(
        self,
        n_wires: int,
        edges: Iterable[tuple[int, int]],
        *,
        path_cache_capacity: int = 4096,
    ) -> None:
        if type(n_wires) is not int:
            raise ValueError("Coupling wire count must be an integer.")
        if n_wires <= 0:
            raise ValueError("Coupling map requires a positive wire count.")
        if type(path_cache_capacity) is not int:
            raise ValueError("Path cache capacity must be an integer.")
        if path_cache_capacity == 1 or path_cache_capacity < 0:
            raise ValueError("Path cache capacity must be zero or at least two.")
        normalized = []
        seen = set()
        for left, right in edges:
            if type(left) is not int or type(right) is not int:
                raise ValueError("Coupling edge endpoints must be integers.")
            if left < 0 or right < 0 or left >= n_wires or right >= n_wires:
                raise ValueError("Coupling edge contains a wire outside the device.")
            if left == right:
                continue
            edge = (min(left, right), max(left, right))
            if edge not in seen:
                normalized.append(edge)
                seen.add(edge)
        object.__setattr__(self, "n_wires", int(n_wires))
        object.__setattr__(self, "edges", tuple(normalized))
        adjacency: list[set[int]] = [set() for _ in range(n_wires)]
        for left, right in normalized:
            adjacency[left].add(right)
            adjacency[right].add(left)
        object.__setattr__(
            self,
            "_adjacency",
            tuple(tuple(sorted(neighbors)) for neighbors in adjacency),
        )
        object.__setattr__(self, "path_cache_capacity", path_cache_capacity)
        object.__setattr__(self, "_path_cache", OrderedDict())
        object.__setattr__(self, "_path_cache_lock", Lock())
        object.__setattr__(self, "_path_cache_hits", 0)
        object.__setattr__(self, "_path_cache_misses", 0)
        object.__setattr__(self, "_path_cache_evictions", 0)
        object.__setattr__(self, "_distance_rows", OrderedDict())
        object.__setattr__(self, "_distance_hits", 0)
        object.__setattr__(self, "_distance_misses", 0)
        object.__setattr__(self, "_distance_evictions", 0)

    @classmethod
    def line(
        cls,
        n_wires: int,
        *,
        path_cache_capacity: int = 4096,
    ) -> "CouplingMap":
        if type(n_wires) is not int:
            raise ValueError("Coupling wire count must be an integer.")
        return cls(
            n_wires,
            ((wire, wire + 1) for wire in range(n_wires - 1)),
            path_cache_capacity=path_cache_capacity,
        )

    @classmethod
    def ring(
        cls,
        n_wires: int,
        *,
        path_cache_capacity: int = 4096,
    ) -> "CouplingMap":
        if type(n_wires) is not int:
            raise ValueError("Coupling wire count must be an integer.")
        edges = [(wire, wire + 1) for wire in range(n_wires - 1)]
        if n_wires > 2:
            edges.append((n_wires - 1, 0))
        return cls(
            n_wires,
            edges,
            path_cache_capacity=path_cache_capacity,
        )

    @classmethod
    def grid(
        cls,
        rows: int,
        cols: int,
        *,
        path_cache_capacity: int = 4096,
    ) -> "CouplingMap":
        if type(rows) is not int or type(cols) is not int:
            raise ValueError("Coupling grid dimensions must be integers.")
        if rows <= 0 or cols <= 0:
            raise ValueError("Coupling grid dimensions must be positive.")
        edges = []
        for row in range(rows):
            for col in range(cols):
                wire = row * cols + col
                if col + 1 < cols:
                    edges.append((wire, wire + 1))
                if row + 1 < rows:
                    edges.append((wire, wire + cols))
        return cls(
            rows * cols,
            edges,
            path_cache_capacity=path_cache_capacity,
        )

    def _validate_wire(self, wire: int) -> int:
        if type(wire) is not int:
            raise ValueError("Coupling wire index must be an integer.")
        if wire < 0 or wire >= self.n_wires:
            raise ValueError(
                f"Coupling wire {wire} is outside [0, {self.n_wires - 1}]."
            )
        return wire

    def neighbors(self, wire: int) -> tuple[int, ...]:
        wire = self._validate_wire(wire)
        return self._adjacency[wire]

    def has_edge(self, left: int, right: int) -> bool:
        left = self._validate_wire(left)
        right = self._validate_wire(right)
        return right in self._adjacency[left]

    def shortest_path(self, start: int, goal: int) -> tuple[int, ...]:
        start = self._validate_wire(start)
        goal = self._validate_wire(goal)
        cache_key = (start, goal)
        with self._path_cache_lock:
            cached = self._path_cache.get(cache_key)
            if cached is not None:
                self._path_cache.move_to_end(cache_key)
                object.__setattr__(self, "_path_cache_hits", self._path_cache_hits + 1)
                return cached
            object.__setattr__(
                self,
                "_path_cache_misses",
                self._path_cache_misses + 1,
            )
        if start == goal:
            identity_path = (start,)
            self._cache_paths((cache_key, identity_path))
            return identity_path
        parents = {start: -1}
        queue: deque[int] = deque([start])
        while queue:
            wire = queue.popleft()
            for neighbor in self.neighbors(wire):
                if neighbor in parents:
                    continue
                parents[neighbor] = wire
                if neighbor == goal:
                    path_nodes = [goal]
                    while path_nodes[-1] != start:
                        path_nodes.append(parents[path_nodes[-1]])
                    resolved = tuple(reversed(path_nodes))
                    self._cache_paths(
                        (cache_key, resolved),
                        ((goal, start), tuple(reversed(resolved))),
                    )
                    return resolved
                queue.append(neighbor)
        raise ValueError(f"No coupling path between wires {start} and {goal}.")

    def _cache_paths(
        self,
        *entries: tuple[tuple[int, int], tuple[int, ...]],
    ) -> None:
        if self.path_cache_capacity == 0:
            return
        with self._path_cache_lock:
            for key, path in entries:
                self._path_cache[key] = path
                self._path_cache.move_to_end(key)
                while len(self._path_cache) > self.path_cache_capacity:
                    self._path_cache.popitem(last=False)
                    object.__setattr__(
                        self,
                        "_path_cache_evictions",
                        self._path_cache_evictions + 1,
                    )

    def distance(self, left: int, right: int) -> int:
        """Return the undirected hop count between two physical wires.

        Raises:
            ValueError: If either wire is outside the device or the two wires
                are not connected by any coupling path.
        """

        left = self._validate_wire(left)
        right = self._validate_wire(right)
        distance = self._distance_row(left)[right]
        if distance == UNREACHABLE_DISTANCE:
            raise ValueError(f"No coupling path between wires {left} and {right}.")
        return distance

    def distance_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Return the dense hop-count matrix over every ordered wire pair.

        Row and column indices are physical wire indices, so entry
        ``matrix[left][right]`` is the undirected distance used by
        :meth:`distance`. Entries equal to :data:`UNREACHABLE_DISTANCE` mean
        the pair has no coupling path. The matrix is symmetric with a zero
        diagonal, and building it costs one breadth-first search per wire.
        """

        return tuple(self._distance_row(source) for source in range(self.n_wires))

    def distance_cache_info(self) -> dict[str, int]:
        """Return bounded-cache diagnostics for the distance index."""

        with self._path_cache_lock:
            return {
                "capacity": self.path_cache_capacity,
                "size": len(self._distance_rows),
                "hits": self._distance_hits,
                "misses": self._distance_misses,
                "evictions": self._distance_evictions,
            }

    def _distance_row(self, source: int) -> tuple[int, ...]:
        """Return the cached all-wire distance row rooted at one wire."""

        with self._path_cache_lock:
            cached = self._distance_rows.get(source)
            if cached is not None:
                self._distance_rows.move_to_end(source)
                object.__setattr__(self, "_distance_hits", self._distance_hits + 1)
                return cached
            object.__setattr__(self, "_distance_misses", self._distance_misses + 1)
        row = _breadth_first_distances(self._adjacency, source)
        if self.path_cache_capacity:
            with self._path_cache_lock:
                self._distance_rows[source] = row
                self._distance_rows.move_to_end(source)
                while len(self._distance_rows) > self.path_cache_capacity:
                    self._distance_rows.popitem(last=False)
                    object.__setattr__(
                        self, "_distance_evictions", self._distance_evictions + 1
                    )
        return row

    def path_cache_info(self) -> dict[str, int]:
        """Return bounded-cache diagnostics for compiler observability."""

        with self._path_cache_lock:
            return {
                "capacity": self.path_cache_capacity,
                "size": len(self._path_cache),
                "hits": self._path_cache_hits,
                "misses": self._path_cache_misses,
                "evictions": self._path_cache_evictions,
            }


@dataclass(frozen=True)
class RoutingCostEstimate:
    """Lightweight routing size estimate without materializing instructions."""

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

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "source_instruction_count": self.source_instruction_count,
            "estimated_instruction_count": self.estimated_instruction_count,
            "topology_gate_count": self.topology_gate_count,
            "routed_gate_count": self.routed_gate_count,
            "planned_inserted_swap_count": self.planned_inserted_swap_count,
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


def _source_routing_counts(
    ir: CircuitIR,
    coupling: CouplingMap,
) -> tuple[int, int, int]:
    """Return the source program's topology, routed, and skipped-channel counts.

    These three describe a program against one device, and every strategy reports
    them the same way: a two-wire operation is a topology gate, one the device
    cannot execute directly is also a routed gate, and a two-wire channel is
    neither because routing may not move a channel. Counting them in one place is
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
        if not coupling.has_edge(*instruction.wires):
            routed_gate_count += 1
    return topology_gate_count, routed_gate_count, skipped_channel_count


def _estimate_swap_plan(
    ir: CircuitIR,
    coupling: CouplingMap,
    *,
    strategy: str,
) -> tuple[int, tuple[int, ...]]:
    """Estimate the SWAPs one estimating strategy inserts, and its mid-plan layout.

    The estimate replays the strategy's layout bookkeeping over the program's wire
    pairs instead of emitting instructions, which is what keeps it cheap.
    ``restore_after_each_gate`` returns to the identity layout after every gate, so
    it pays each hop twice and ends where it started; ``persistent_layout`` keeps
    the layout it reached, so it pays each hop once and reports where it finished.

    Returns:
        ``(planned_inserted_swap_count, pre_restore_logical_to_physical)``.
    """

    logical_to_physical = list(range(ir.n_wires))
    physical_to_logical = list(range(ir.n_wires))
    forward_swap_count = 0
    for instruction in ir:
        if len(instruction.wires) != 2:
            continue
        if instruction.metadata.get("is_channel"):
            continue
        mapped_wires = tuple(logical_to_physical[wire] for wire in instruction.wires)
        if coupling.has_edge(*mapped_wires):
            continue
        path = coupling.shortest_path(*mapped_wires)
        if any(wire >= ir.n_wires for wire in path):
            raise ValueError(
                "Routing through physical ancilla wires outside the circuit IR "
                "is not supported; provide an explicit layout/lowering step."
            )
        gate_swap_count = max(0, len(path) - 2)
        forward_swap_count += gate_swap_count
        if strategy == "persistent_layout":
            for path_index in range(gate_swap_count):
                left = path[path_index]
                right = path[path_index + 1]
                left_logical = physical_to_logical[left]
                right_logical = physical_to_logical[right]
                physical_to_logical[left], physical_to_logical[right] = (
                    right_logical,
                    left_logical,
                )
                logical_to_physical[left_logical] = right
                logical_to_physical[right_logical] = left
    identity_layout = tuple(range(ir.n_wires))
    pre_restore_layout = (
        tuple(logical_to_physical)
        if strategy == "persistent_layout"
        else identity_layout
    )
    return 2 * forward_swap_count, pre_restore_layout


def _undirected_map(coupling_map: object, wire_count: int, entry: str) -> CouplingMap:
    """Coerce an edge sequence; refuse a directed device by name.

    ``DirectedCouplingMap`` is not iterable, so the fallback used to fail with the
    iteration's own ``TypeError`` and named neither ``legalize_circuit_topology``
    nor the two forms this accepts.
    """

    if isinstance(coupling_map, CouplingMap):
        return coupling_map
    if not isinstance(coupling_map, Iterable):
        raise TypeError(
            f"{entry} takes a CouplingMap or an iterable of (control, target) edges, "
            f"not a {type(coupling_map).__name__}; a directed coupling map carries CX "
            "direction this undirected cost model does not read, so route it through "
            "flagquantum.compiler.legalize_circuit_topology"
        )
    return CouplingMap(wire_count, coupling_map)


def estimate_routing_cost(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | Iterable[tuple[int, int]],
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

    Args:
        circuit_or_ir: The program to price.
        coupling_map: The device to price it against, as a ``CouplingMap`` or an
            edge sequence.
        strategy: One of ``ROUTING_STRATEGIES``.

    Returns:
        The planned SWAP count, the expected instruction count, and the source
        counts the same strategy would report in routing metadata.

    Raises:
        TypeError: If ``coupling_map`` is neither a ``CouplingMap`` nor an edge sequence.
        ValueError: If ``strategy`` is not in ``ROUTING_STRATEGIES``, the coupling
            map has fewer wires than the circuit, a costed path leaves the
            device, or a planner cannot route the program.
    """

    if strategy not in ROUTING_STRATEGIES:
        raise ValueError(
            "routing strategy must be one of "
            + ", ".join(repr(name) for name in ROUTING_STRATEGIES)
        )
    ir = ensure_circuit_ir(circuit_or_ir)
    coupling = _undirected_map(coupling_map, ir.n_wires, "estimate_routing_cost")
    if coupling.n_wires < ir.n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")

    identity_layout = tuple(range(ir.n_wires))
    cache_before = coupling.path_cache_info()
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, coupling)
    )
    if strategy in SABRE_ROUTING_STRATEGIES:
        initial_layout = (
            plan_sabre_layout(ir, coupling)
            if strategy == "sabre_layout"
            else identity_layout
        )
        plan = plan_sabre_swaps(ir, coupling, initial_layout=initial_layout)
        planned_swaps = len(plan.swaps) + len(plan_restore_swaps(plan, coupling))
        pre_restore_layout = plan.final_logical_to_physical
    else:
        planned_swaps, pre_restore_layout = _estimate_swap_plan(
            ir, coupling, strategy=strategy
        )
    cache_after = coupling.path_cache_info()
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
    )


def select_routing_strategy(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | Iterable[tuple[int, int]],
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

    Raises:
        TypeError: If ``coupling_map`` is neither a ``CouplingMap`` nor an edge sequence.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    coupling = _undirected_map(coupling_map, ir.n_wires, "select_routing_strategy")
    candidates = {
        name: estimate_routing_cost(ir, coupling, strategy=name)
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
    coupling: CouplingMap,
    strategy: str,
    topology_gate_count: int,
    routed_gate_count: int,
    inserted_swap_count: int,
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
        "coupling_n_wires": coupling.n_wires,
        "coupling_edges": coupling.edges,
        "initial_logical_to_physical": initial_layout,
        "pre_restore_logical_to_physical": pre_restore_layout,
        "final_logical_to_physical": identity_layout,
        "mapping_restored": True,
        "direction_semantics": "logical_wire_order_preserved",
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


def _require_multi_wire_device_local(
    instruction: Instruction,
    wires: tuple[int, ...],
    has_edge: Callable[[int, int], bool],
) -> None:
    """Fail closed unless a multi-wire instruction is executable on the device.

    No router in this package synthesizes a three-or-more-wire instruction onto
    physical couplings, and a routing strategy cannot insert SWAPs for one, so a
    multi-wire instruction has to be decomposable where it stands. The required
    couplings are read from `_MULTI_WIRE_OPERAND_PAIRS` rather than inferred from
    the operand order, because the two known multi-wire instructions do not
    interact on consecutive operands; an instruction with no entry is refused
    instead of assumed local. Two-wire instructions keep their existing
    SWAP-based handling.
    """

    if len(wires) < 3:
        return
    required = _MULTI_WIRE_OPERAND_PAIRS.get(instruction.name)
    if required is None:
        raise ValueError(
            f"multi-wire instruction {instruction.name!r} has no verified physical "
            f"connectivity rule: physical wires {wires}; decompose multi-wire "
            "instructions onto device couplings before routing"
        )
    missing = tuple(
        (wires[left], wires[right])
        for left, right in required
        if not has_edge(wires[left], wires[right])
    )
    if missing:
        raise ValueError(
            f"multi-wire instruction {instruction.name!r} requires physical "
            f"couplings {missing} the device does not carry: physical wires "
            f"{wires}; decompose multi-wire instructions onto device couplings "
            "before routing"
        )


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
    coupling: CouplingMap,
    *,
    path_cache_before: dict[str, int],
) -> CircuitIR:
    strategy = "persistent_layout"
    logical_to_physical = list(range(ir.n_wires))
    physical_to_logical = list(range(ir.n_wires))
    routed: list[Instruction] = []
    routing_swaps: list[tuple[int, int, Instruction, int]] = []
    # Counted against the source wires, like every other strategy and like the
    # cost estimate: a gate this layout happened to make local is still a gate
    # the device did not carry, and reporting it as unrouted would make the same
    # field mean something different per strategy.
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, coupling)
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
                instruction, mapped_wires, coupling.has_edge
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
        if not coupling.has_edge(left, right):
            path = coupling.shortest_path(left, right)
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
        coupling=coupling,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=2 * len(routing_swaps),
        skipped_channel_count=skipped_channel_count,
        initial_layout=tuple(range(ir.n_wires)),
        pre_restore_layout=pre_restore_layout,
        path_cache_before=path_cache_before,
        path_cache_after=coupling.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def _route_sabre(
    ir: CircuitIR,
    coupling: CouplingMap,
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
    """

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
        _source_routing_counts(ir, coupling)
    )
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
                instruction, placed_wires, coupling.has_edge
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
        routed.append(
            _remap_instruction(
                instruction,
                placed_wires,
                strategy=strategy,
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
        coupling=coupling,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=len(plan.swaps) + len(restore),
        skipped_channel_count=skipped_channel_count,
        initial_layout=initial_layout,
        pre_restore_layout=plan.final_logical_to_physical,
        path_cache_before=path_cache_before,
        path_cache_after=coupling.path_cache_info(),
    )
    return replace(ir, instructions=tuple(routed), metadata=metadata)


def route_to_topology(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | Iterable[tuple[int, int]],
    *,
    strategy: str = "restore_after_each_gate",
) -> CircuitIR:
    """Insert SWAP gates so two-qubit operations respect hardware topology.

    The device is undirected; ``legalize_circuit_topology`` is the entry that reads
    direction, and a directed device is refused here rather than routed against its
    undirected projection.

    Raises:
        TypeError: If ``coupling_map`` is neither a ``CouplingMap`` nor an edge sequence.
        ValueError: If ``strategy`` is not in ``ROUTING_STRATEGIES``.
    """

    if strategy not in ROUTING_STRATEGIES:
        raise ValueError(
            "routing strategy must be one of "
            + ", ".join(repr(name) for name in ROUTING_STRATEGIES)
        )
    ir = ensure_circuit_ir(circuit_or_ir)
    coupling = _undirected_map(coupling_map, ir.n_wires, "route_to_topology")
    if coupling.n_wires < ir.n_wires:
        raise ValueError("Coupling map has fewer wires than the circuit.")
    path_cache_before = coupling.path_cache_info()
    if strategy == "persistent_layout":
        return _route_persistent_layout(
            ir,
            coupling,
            path_cache_before=path_cache_before,
        )
    if strategy in SABRE_ROUTING_STRATEGIES:
        return _route_sabre(
            ir,
            coupling,
            strategy=strategy,
            path_cache_before=path_cache_before,
        )

    routed: list[Instruction] = []
    topology_gate_count, routed_gate_count, skipped_channel_count = (
        _source_routing_counts(ir, coupling)
    )
    inserted_swap_count = 0
    for source_index, instruction in enumerate(ir):
        if len(instruction.wires) != 2:
            _require_multi_wire_device_local(
                instruction, instruction.wires, coupling.has_edge
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

        left, right = instruction.wires
        if coupling.has_edge(left, right):
            routed.append(
                _remap_instruction(
                    instruction,
                    instruction.wires,
                    strategy=strategy,
                    source_instruction_index=source_index,
                )
            )
            continue

        path = coupling.shortest_path(left, right)
        if any(wire >= ir.n_wires for wire in path):
            raise ValueError(
                "Routing through physical ancilla wires outside the circuit IR "
                "is not supported; provide an explicit layout/lowering step."
            )
        forward_swaps = [
            (path[index], path[index + 1]) for index in range(len(path) - 2)
        ]
        inserted_swap_count += 2 * len(forward_swaps)
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
        routed.append(
            Instruction(
                name=instruction.name,
                wires=(path[-2], path[-1]),
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
        coupling=coupling,
        strategy=strategy,
        topology_gate_count=topology_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=inserted_swap_count,
        skipped_channel_count=skipped_channel_count,
        initial_layout=tuple(range(ir.n_wires)),
        pre_restore_layout=tuple(range(ir.n_wires)),
        path_cache_before=path_cache_before,
        path_cache_after=coupling.path_cache_info(),
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
