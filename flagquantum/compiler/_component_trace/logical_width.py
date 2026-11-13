# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QubitCount.compute.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py

"""Query recursive logical widths through native decomposition owners."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence

from .call_graph import Generalizer, Node, compose_cost_generalizer
from .cost_query import _query_cost
from .symbolic_math import SymbolicInt, smax
from .width_schedule import _validate_width


def query_logical_width(
    root: Node,
    *,
    signature_width: Callable[[Node], SymbolicInt],
    callees: Callable[[Node], Iterable[tuple[Node, SymbolicInt]]],
    decomposition_width: (
        Callable[[Node, Callable[[Node], SymbolicInt]], SymbolicInt | None] | None
    ) = None,
    cache: MutableMapping[Node, SymbolicInt] | None = None,
    static_costs: Mapping[Node, SymbolicInt] | None = None,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
) -> SymbolicInt:
    """Prefer connected decomposition width, then signature/callee maximum.

    The decomposition owner receives the recursive child query. It can feed
    those child widths into ``estimate_connection_width`` or the existing
    register-program estimator. Return ``None`` only when decomposition is
    unavailable. A computed zero is authoritative. Errors propagate.

    The fallback follows the source model and may underestimate simultaneous
    workspace. It visits every declared callee, including zero multiplicities.
    Widths are peaks and are independent of call repetition counts.

    The caller-owned cache contains only visited nodes. Pass ``cache.get`` to
    the resource call-graph exporter to preserve absent width rows. Reuse a
    cache only with the same owners, overrides and generalizer.

    Examples:
        >>> widths: dict[str, SymbolicInt] = {}
        >>> query_logical_width(
        ...     "root", signature_width=lambda node: 3,
        ...     callees=lambda node: (), cache=widths,
        ... )
        3
        >>> widths
        {'root': 3}
    """

    def compute(node: Node, child: Callable[[Node], SymbolicInt]) -> SymbolicInt:
        width = (
            None if decomposition_width is None else decomposition_width(node, child)
        )
        if width is not None:
            _validate_width(width)
            return width
        width = signature_width(node)
        _validate_width(width)
        for callee, _ in callees(node):
            width = smax(width, child(callee))
        return width

    return _query_cost(
        root, compute, 0, compose_cost_generalizer(generalizer), cache, static_costs
    )
