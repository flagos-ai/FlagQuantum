"""Undirected hardware connectivity used by the routing passes.

A coupling map states which physical wire pairs a device couples, without saying
which way round a two-wire operation may run across them. The ordered case is a
different type in a module beside this one, so a reader asking what a device
permits reads one of two files rather than searching one file for the branch that
decides it.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from threading import Lock

# Dense distance-matrix entry for a physical wire pair with no coupling path.
# A coupling map is a graph, so most pairs of a production device are far
# apart but still reachable; this value is reserved for genuine disconnection.
UNREACHABLE_DISTANCE = -1


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


__all__ = [
    "UNREACHABLE_DISTANCE",
    "CouplingMap",
]
