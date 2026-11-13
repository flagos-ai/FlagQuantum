# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 flame_graph._populate_flame_graph_data.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/flame_graph.py

"""Stream declared T-cost paths in the folded-stack FlameGraph format."""

from collections.abc import Callable, Generator, Iterator, Mapping

from ..compiler._component_trace.call_graph import Node
from .call_graph import _call_order


def iter_flame_graph_data(
    root: Node,
    direct_calls: Mapping[tuple[Node, Node], int],
    t_counts: Mapping[Node, int],
    *,
    title: Callable[[Node], str] = str,
    is_t_gate: Callable[[Node], bool] | None = None,
) -> Generator[str, None, None]:
    """Yield DFS paths, repeating complete subtrees by call multiplicity.

    Counts describe full subtrees even when direct_calls is depth-limited.
    A single positive-cost T or T-adjoint child is represented by its parent
    frame when is_t_gate identifies it. Zero-cost children are omitted.
    The iterator keeps graph and path memory, independent of emitted row count.
    Validation runs before the first row. Cycles, missing or nonconcrete costs,
    negative multiplicities and frame separators in titles raise ValueError.

    Examples:
        >>> list(iter_flame_graph_data(
        ...     "Root", {("Root", "And"): 2}, {"Root": 8, "And": 4}
        ... ))
        ['Root(T:8);And(T:4)\\t4', 'Root(T:8);And(T:4)\\t4']
    """
    for count in direct_calls.values():
        if type(count) is not int or count < 0:
            raise ValueError("Flame graph multiplicities must be nonnegative integers")
    order, successors = _call_order(root, direct_calls)
    frames = {}
    for node in order:
        cost = t_counts.get(node)
        if type(cost) is not int or cost < 0:
            raise ValueError("Flame graph T costs must be nonnegative integers")
        label = title(node)
        if any(separator in label for separator in (";", "\t", "\n", "\r")):
            raise ValueError("Flame graph titles cannot contain frame separators")
        frames[node] = f"{label}(T:{cost})"
    path: list[str] = []

    def visit(node: Node) -> Iterator[str]:
        children = [child for child in successors.get(node, ()) if t_counts[child] > 0]
        path.append(frames[node])
        if not children or (
            len(children) == 1 and is_t_gate is not None and is_t_gate(children[0])
        ):
            yield ";".join(path) + "\t" + str(t_counts[node])
        else:
            for child in children:
                for _ in range(direct_calls[node, child]):
                    yield from visit(child)
        path.pop()

    yield from visit(root)
