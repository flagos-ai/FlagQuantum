"""Small graph primitives used by the native compiler."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Hashable, Iterable


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}

    def find(self, item: Hashable) -> Hashable:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: Hashable, right: Hashable) -> None:
        self.parent[self.find(right)] = self.find(left)


class DiGraph:
    def __init__(self) -> None:
        self._out: dict[Hashable, set[Hashable]] = defaultdict(set)
        self._in: dict[Hashable, set[Hashable]] = defaultdict(set)

    def add_edge(self, source: Hashable, target: Hashable) -> None:
        self._out[source].add(target)
        self._in[target].add(source)
        self._out.setdefault(target, set())
        self._in.setdefault(source, set())

    def nodes(self) -> set[Hashable]:
        return set(self._out) | set(self._in)

    def successors(self, node: Hashable) -> set[Hashable]:
        return set(self._out.get(node, ()))

    def topological_layers(self) -> list[list[Hashable]]:
        indegree = {node: len(self._in[node]) for node in self.nodes()}
        ready = deque(node for node, degree in indegree.items() if degree == 0)
        layers: list[list[Hashable]] = []
        while ready:
            layer = list(ready)
            ready.clear()
            layers.append(layer)
            for node in layer:
                for succ in self._out[node]:
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        ready.append(succ)
        if sum(len(layer) for layer in layers) != len(indegree):
            raise ValueError("Graph contains a cycle.")
        return layers


def connected_components(
    edges: Iterable[tuple[Hashable, Hashable]],
) -> list[set[Hashable]]:
    uf = UnionFind()
    nodes = set()
    for left, right in edges:
        nodes.update((left, right))
        uf.union(left, right)
    groups: dict[Hashable, set[Hashable]] = defaultdict(set)
    for node in nodes:
        groups[uf.find(node)].add(node)
    return list(groups.values())


__all__ = ["UnionFind", "DiGraph", "connected_components"]
