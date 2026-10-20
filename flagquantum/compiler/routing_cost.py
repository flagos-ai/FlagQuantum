"""The device facts a routing cost model reads, and the coercion that builds one.

Routing prices a program against a device before it emits anything: how many
physical qubits the device has, which couplings it declares, whether a placed
gate can execute where it stands, and -- when it cannot -- which physical path to
move it along. Those four questions are the same on an undirected and on an
ordered device, so they are answered once here and read by the estimators and by
the planners alike.

This is a module of its own because the questions are the model's rather than the
router's: ``flagquantum/compiler/routing.py`` emits instructions and decides where
they stand, and it is long enough that the device introspection it prices against
has to be able to grow without displacing the routes it sits beside.

A device may also state what each of its links costs to use.
``LinkErrorRates`` is that statement: a two-qubit error probability per coupling,
which turns the one question a connectivity-only device cannot answer -- which of
two legal routes to take -- into a question with an answer that is not a hop
count.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from heapq import heappop, heappush
from threading import Lock

from ..core.ir import Instruction
from .coupling import CouplingMap
from .directed_topology import DirectedCouplingMap
from .operand_semantics import _operand_symmetry
from .ordering_barrier import is_ordering_barrier


class _TopologyCostModel:
    """The device facts a routing cost model reads, ordered or unordered.

    A cost model asks its device four things: how many physical qubits it has,
    which couplings it declares, whether a two-qubit instruction can execute where
    it stands, and -- when it cannot -- which physical path to move it along. On an
    undirected ``CouplingMap`` the third question is the second one, because
    either direction of a link satisfies every two-qubit instruction. On a
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
    link is legal in either direction; and multi-qubit locality is checked on weak
    edges, because a three-qubit instruction has no operand order to satisfy on the
    couplings it spans. Direction decides whether a placed gate is executable, not
    whether the router may cross the link.

    A device that also states a per-link error rate answers the path question
    differently, and the model carries that calibration rather than replacing
    itself: ``path`` returns the lowest-error route when the device is calibrated
    and the shortest route when it is not, so one planner serves both devices and a
    caller can read which question was answered from ``link_error_rates``.
    """

    __slots__ = ("_device", "_weak", "_link_error_rates")

    def __init__(
        self,
        device: CouplingMap | DirectedCouplingMap,
        link_error_rates: LinkErrorRates | None = None,
    ) -> None:
        self._device = device
        self._weak = (
            device
            if isinstance(device, CouplingMap)
            else CouplingMap(device.n_qubits, device.edges)
        )
        if link_error_rates is not None and (
            link_error_rates.n_qubits != device.n_qubits
            or tuple(link_error_rates.edges) != tuple(device.edges)
        ):
            raise ValueError(
                "link error rates describe a different device than the couplings "
                "being priced"
            )
        self._link_error_rates = link_error_rates

    @property
    def n_qubits(self) -> int:
        return self._device.n_qubits

    @property
    def link_error_rates(self) -> LinkErrorRates | None:
        """Return the device calibration this model routes against, if any."""

        return self._link_error_rates

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
        if self._link_error_rates is not None:
            return self._link_error_rates.path_cache_info()
        return self._weak.path_cache_info()

    def weak_edge(self, left: int, right: int) -> bool:
        return self._weak.has_edge(left, right)

    def path(self, wires: tuple[int, ...]) -> tuple[int, ...]:
        """Return the qubit sequence a SWAP series must follow to join two qubits.

        The shortest such sequence when the device states only its couplings. When
        it also states a per-link error rate, the sequence that pays the least
        summed link error, which may be a longer sequence than the shortest one:
        that substitution is the whole difference between pricing a route in hops
        and pricing it in faults.
        """

        if self._link_error_rates is not None:
            return self._link_error_rates.lowest_error_path(*wires)
        return self._weak.shortest_path(*wires)

    def link_error(self, left: int, right: int) -> float:
        """Return one link's two-qubit error probability.

        Raises:
            ValueError: If the device states no link error rate, so there is no
                such quantity to read. Returning zero would price every link as
                faultless and silently turn a weighted route into a shortest route.
        """

        if self._link_error_rates is None:
            raise ValueError(
                "this device states no link error rate, so it cannot be priced in "
                "faults: pass a LinkErrorRates device to weigh a route"
            )
        return self._link_error_rates.error(left, right)

    def executes(self, instruction: Instruction, wires: tuple[int, ...]) -> bool:
        """Report whether the device runs this instruction on these qubits.

        Raises:
            ValueError: A directed device is asked about a two-qubit opcode with no
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
            return self._weak.has_edge(control, target)
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


@dataclass(frozen=True)
class LinkErrorRates:
    """A device's couplings together with the two-qubit error of each link.

    A ``CouplingMap`` states which qubit pairs a device couples and says nothing
    about how well it couples them, so the cheapest legal route and the cleanest
    legal route are the same route: a hop count. These are the same couplings with
    one measured number per link, and the number is the device fact that separates
    the two questions.

    What the number means is stated rather than assumed, because two routers that
    disagreed about it would choose different routes from the same calibration.
    Each link carries an independent two-qubit error probability ``e`` in ``[0, 1)``
    -- the chance that a two-qubit operation run across that link faults -- and the
    cost of a route is the sum of the ``e`` of the links it uses. A sum of
    probabilities is the expected number of fault events rather than a probability
    of any one of them, which is exactly what makes it additive along a route and
    comparable between two routes of different lengths. Correlated and
    gate-dependent error is not expressible here: one link carries one number,
    whatever runs on it.

    The record refuses a calibration it cannot use rather than filling a gap with a
    default. Every coupling the device declares needs a rate, because a link
    priced at nothing looks like a free route; every rate needs a declared
    coupling, because a rate on a link that does not exist describes a device
    nobody has. Both refusals name the link.

    Examples:
        >>> rates = LinkErrorRates(
        ...     CouplingMap(3, [(0, 1), (1, 2)]),
        ...     {(0, 1): 0.04, (1, 2): 0.0005},
        ... )
        >>> rates.lowest_error_path(0, 2)
        (0, 1, 2)
        >>> rates.error(1, 2)
        0.0005
    """

    device: CouplingMap
    link_errors: tuple[tuple[int, int, float], ...]
    path_cache_capacity: int = field(default=4096, compare=False, hash=False)
    _adjacency: tuple[tuple[int, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _log_weights: dict[tuple[int, int], float] = field(
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

    def __init__(
        self,
        device: CouplingMap,
        link_errors: Mapping[tuple[int, int], float],
        *,
        path_cache_capacity: int = 4096,
    ) -> None:
        if isinstance(device, DirectedCouplingMap):
            raise ValueError(
                "link error rates describe an undirected device: an ordered "
                "device prices a wrong-way gate in SWAPs rather than in link "
                "error, so one rate per link would leave that repair unpriced"
            )
        if not isinstance(device, CouplingMap):
            raise TypeError(
                "link error rates require a CouplingMap, not a "
                f"{type(device).__name__}"
            )
        if type(path_cache_capacity) is not int:
            raise ValueError("Path cache capacity must be an integer.")
        if path_cache_capacity == 1 or path_cache_capacity < 0:
            raise ValueError("Path cache capacity must be zero or at least two.")
        normalized: dict[tuple[int, int], float] = {}
        for link, rate in link_errors.items():
            if not isinstance(link, tuple) or len(link) != 2:
                raise ValueError(
                    "link error rates are keyed by a (qubit, qubit) pair, not by "
                    f"{link!r}"
                )
            left, right = link
            if type(left) is not int or type(right) is not int:
                raise ValueError("link error rate endpoints must be integers.")
            if left == right:
                raise ValueError("a link joins two different qubits, not one.")
            if not device.has_edge(left, right):
                raise ValueError(
                    f"link ({left}, {right}) is not a coupling the device declares"
                )
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ValueError(
                    f"the error rate of link ({left}, {right}) must be a number"
                )
            probability = float(rate)
            if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
                raise ValueError(
                    f"the error rate of link ({left}, {right}) must be finite and "
                    f"in [0, 1), not {probability!r}: a rate of 1 or more describes "
                    "a link no route can use"
                )
            normalized[(min(left, right), max(left, right))] = probability
        missing = [pair for pair in device.edges if pair not in normalized]
        if missing:
            left, right = missing[0]
            raise ValueError(
                f"the device declares link ({left}, {right}) with no error rate; "
                "every coupling needs one, because a link priced at nothing is a "
                "free route"
            )
        adjacency: list[set[int]] = [set() for _ in range(device.n_qubits)]
        for left, right in device.edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        object.__setattr__(self, "device", device)
        object.__setattr__(
            self,
            "link_errors",
            tuple(
                (left, right, normalized[(left, right)]) for left, right in device.edges
            ),
        )
        object.__setattr__(self, "path_cache_capacity", path_cache_capacity)
        object.__setattr__(
            self,
            "_adjacency",
            tuple(tuple(sorted(neighbors)) for neighbors in adjacency),
        )
        # A route's cost is a sum, and the number that sums along a route is the
        # negative logarithm of the probability the route survives its links.
        # Turning the product into a sum is what lets one shortest-path search
        # price a route at all; a rate of zero has weight zero and is a legal edge.
        object.__setattr__(
            self,
            "_log_weights",
            {
                pair: -math.log1p(-probability)
                for pair, probability in normalized.items()
            },
        )
        object.__setattr__(self, "_path_cache", OrderedDict())
        object.__setattr__(self, "_path_cache_lock", Lock())
        object.__setattr__(self, "_path_cache_hits", 0)
        object.__setattr__(self, "_path_cache_misses", 0)
        object.__setattr__(self, "_path_cache_evictions", 0)

    @property
    def n_qubits(self) -> int:
        return self.device.n_qubits

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return the couplings the calibrated device declares."""

        return self.device.edges

    def error(self, left: int, right: int) -> float:
        """Return the two-qubit error probability of one declared link."""

        pair = (min(left, right), max(left, right))
        for candidate_left, candidate_right, rate in self.link_errors:
            if (candidate_left, candidate_right) == pair:
                return rate
        raise ValueError(f"link ({left}, {right}) carries no error rate")

    def route_error(self, qubits: tuple[int, ...]) -> float:
        """Return the expected two-qubit faults one route across these qubits pays.

        Counted as one operation per consecutive link of the route, including the
        last one, which is the link the gate itself runs on. A strategy that pays a
        hop twice -- ``restore_after_each_gate`` restores the layout after every
        gate -- pays that link's error twice in the program it emits, so this is
        the route's cost and not the program's.
        """

        return sum(
            self.error(qubits[index], qubits[index + 1])
            for index in range(len(qubits) - 1)
        )

    def lowest_error_path(self, start: int, goal: int) -> tuple[int, ...]:
        """Return the route between two qubits that pays the least link error.

        The search minimises the route's summed link error, then its hop count,
        then its lexical order, so the answer is a property of the pair rather than
        of the order the question was asked in: the search always runs from the
        lower-numbered endpoint and the result is reversed when the caller asked
        the other way round. Fewest hops is the second key rather than the first,
        which is what keeps a route from growing longer unless it buys a strictly
        smaller error for doing so.

        Raises:
            ValueError: If either qubit is not on the device, or the device has no
                coupling path between them.
        """

        for qubit in (start, goal):
            if type(qubit) is not int or not 0 <= qubit < self.n_qubits:
                raise ValueError(
                    f"physical qubit {qubit!r} is not on this device of "
                    f"{self.n_qubits} qubits"
                )
        if start == goal:
            return (start,)
        first, second = (start, goal) if start < goal else (goal, start)
        cache_key = (first, second)
        with self._path_cache_lock:
            cached = self._path_cache.get(cache_key)
            if cached is not None:
                self._path_cache.move_to_end(cache_key)
                object.__setattr__(self, "_path_cache_hits", self._path_cache_hits + 1)
                return cached if start < goal else tuple(reversed(cached))
            object.__setattr__(self, "_path_cache_misses", self._path_cache_misses + 1)
        resolved = self._search(first, second)
        self._store_path(cache_key, resolved)
        return resolved if start < goal else tuple(reversed(resolved))

    def _search(self, start: int, goal: int) -> tuple[int, ...]:
        frontier: list[tuple[float, int, tuple[int, ...]]] = [(0.0, 0, (start,))]
        settled: dict[int, tuple[float, int, tuple[int, ...]]] = {
            start: (0.0, 0, (start,))
        }
        while frontier:
            cost, hops, path = heappop(frontier)
            qubit = path[-1]
            if qubit == goal:
                return path
            if settled.get(qubit) != (cost, hops, path):
                continue
            for neighbor in self._adjacency[qubit]:
                candidate = (
                    _quantized(
                        cost
                        + self._log_weights[
                            (min(qubit, neighbor), max(qubit, neighbor))
                        ]
                    ),
                    hops + 1,
                    path + (neighbor,),
                )
                incumbent = settled.get(neighbor)
                if incumbent is None or candidate < incumbent:
                    settled[neighbor] = candidate
                    heappush(frontier, candidate)
        raise ValueError(f"No coupling path between qubits {start} and {goal}.")

    def _store_path(
        self,
        cache_key: tuple[int, int],
        path: tuple[int, ...],
    ) -> None:
        if self.path_cache_capacity == 0:
            return
        with self._path_cache_lock:
            self._path_cache[cache_key] = path
            self._path_cache.move_to_end(cache_key)
            while len(self._path_cache) > self.path_cache_capacity:
                self._path_cache.popitem(last=False)
                object.__setattr__(
                    self, "_path_cache_evictions", self._path_cache_evictions + 1
                )

    def path_cache_info(self) -> dict[str, int]:
        with self._path_cache_lock:
            return {
                "capacity": self.path_cache_capacity,
                "size": len(self._path_cache),
                "hits": self._path_cache_hits,
                "misses": self._path_cache_misses,
                "evictions": self._path_cache_evictions,
            }


#: The device forms a routing entry prices: a declared coupling map, an ordered
#: one, a calibrated one, an edge sequence, or the cost model this module builds
#: from them. The last member is what makes one weak projection and one path cache
#: reusable across the strategies an automatic choice prices.
_PricedDevice = (
    CouplingMap
    | DirectedCouplingMap
    | LinkErrorRates
    | Iterable[tuple[int, int]]
    | _TopologyCostModel
)


def require_weight_reading_strategy(
    strategy: str,
    topology: _TopologyCostModel,
    entry: str,
    reading_strategies: tuple[str, ...],
) -> None:
    """Refuse a strategy that would route a calibrated device without reading it.

    A caller who calibrates a device has stated which of its links are clean and
    which are not. A strategy that chooses routes by hop count answers a different
    question, so running it would report a plan the calibration had no part in and
    leave the caller to assume otherwise. The refusal names both sides and the
    strategies that can be used instead, because the repair is a choice the caller
    makes rather than one this function can make for it.

    `reading_strategies` is the caller's vocabulary rather than this module's: the
    router owns which strategies it has and which of them read a weight, and this
    function owns the check.
    """

    if topology.link_error_rates is None:
        return
    if strategy in reading_strategies:
        return
    raise ValueError(
        f"{entry} cannot route a device that states a per-link error rate with "
        f"strategy {strategy!r}: that strategy chooses its routes by hop count and "
        f"would ignore the calibration. Use one of "
        + ", ".join(repr(name) for name in reading_strategies)
        + "."
    )


def emitted_link_faults(
    routed: Iterable[Instruction],
    topology: _TopologyCostModel,
) -> float | None:
    """Sum the declared link error of every two-qubit operation a routed program runs.

    Read off the program rather than accumulated while emitting it, so the number
    describes what a device would have to execute: every SWAP the strategy inserted,
    every routed gate on the link it landed on, and nothing that is not a
    two-qubit device operation. A channel instruction spans two qubits without
    being a gate, and an ordering barrier carries no qubits of its own, so both are
    excluded -- the same exclusions the source counts make, which is why the two
    totals describe the same program.

    ``None`` when the device states no per-link error rate, in which case there is
    no such quantity to report.
    """

    if topology.link_error_rates is None:
        return None
    total = 0.0
    for instruction in routed:
        if len(instruction.wires) != 2:
            continue
        if instruction.metadata.get("is_channel"):
            continue
        if is_ordering_barrier(instruction):
            continue
        total += topology.link_error(*instruction.wires)
    return total


def _quantized(cost: float) -> float:
    """Round a route cost to the precision two routes are compared at.

    A route's cost is a sum of logarithms, so two routes of equal exact cost can
    differ in the last digits according to the order the terms were added. Left
    alone, that difference is the first key of the search, so the longer of two
    equally clean routes would win by an accident of floating-point arithmetic
    rather than the hop count deciding it. Twelve decimal places is far coarser
    than any difference a calibration distinguishes -- an error rate of ``1e-13``
    per link is below what a device reports -- and far finer than the hop-count
    gap between routes that genuinely differ.
    """

    return round(cost, 12)


def _topology_cost_model(
    coupling_map: object,
    wire_count: int,
    entry: str,
) -> _TopologyCostModel:
    """Coerce the device a cost-model entry was handed.

    A directed device is a device the cost model reads, not one it refuses: the
    ordered edges decide which placed gates need a reversal, and the weak
    projection decides where the SWAPs go.

    A calibrated device is the same couplings with a per-link error rate beside
    them, so the model keeps both: the couplings answer executability, and the
    calibration answers which route to take.

    A model is returned unchanged so that a caller pricing several strategies --
    ``select_routing_strategy`` prices three -- builds one weak projection and one
    path cache, and so measures a later candidate's cache traffic against the
    earlier candidates' work rather than against a cold device.
    """

    if isinstance(coupling_map, _TopologyCostModel):
        return coupling_map
    if isinstance(coupling_map, LinkErrorRates):
        return _TopologyCostModel(
            coupling_map.device,
            link_error_rates=coupling_map,
        )
    if isinstance(coupling_map, (CouplingMap, DirectedCouplingMap)):
        return _TopologyCostModel(coupling_map)
    if not isinstance(coupling_map, Iterable):
        raise TypeError(
            f"{entry} takes a CouplingMap, a DirectedCouplingMap, a LinkErrorRates "
            f"device, or an iterable of (control, target) edges, not a "
            f"{type(coupling_map).__name__}"
        )
    return _TopologyCostModel(CouplingMap(wire_count, coupling_map))
