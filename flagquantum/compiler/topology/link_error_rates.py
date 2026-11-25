"""Calibrated link errors, deterministic weighted paths and bounded path caching."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from heapq import heappop, heappush
from threading import Lock

from .coupling import CouplingMap
from .directed_topology import DirectedCouplingMap


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


# Preserve the established class reference used by serialized type lookups.
LinkErrorRates.__module__ = "flagquantum.compiler.topology.routing_cost"


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
