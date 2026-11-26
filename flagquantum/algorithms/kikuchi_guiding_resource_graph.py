# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Kikuchi guiding-state tutorial and its
# direct build_call_graph dependencies at bcb17bb25360ed68ce3bc084645db3e929799389.

"""Bounded source-model diagrams composed from domain-owned declarations."""

from collections.abc import Hashable

from ..compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ..compiler.registers.comparison.half_ordering import (
    source_unsigned_half_less_than_calls,
)
from ..compiler.registers.comparison.ordering_declaration import ComparatorDeclaration
from ..compiler.registers.sorting.in_place_sort_declaration import (
    source_in_place_sort_calls,
)
from ..compiler.registers.sorting.sorted_duplicates_resource_model import (
    source_unsigned_sorted_duplicates_calls,
)
from ..core.value_math import SymbolicInt
from .kikuchi_guiding_state import GuidingState, ProbabilisticUncompute


def source_guiding_call_graph(
    guide: GuidingState, *, max_depth: int = 1
) -> SymbolicCallGraph[Hashable]:
    """Retain the tutorial's source call hierarchy through depth three.

    This is a logical model diagram, not a physical decomposition or a gate
    total. Depth one admits symbolic metadata; deeper diagrams require a
    concrete instance. Leaves at depth three deliberately remain unexpanded.
    """
    if type(max_depth) is not int or not 0 <= max_depth <= 3:
        raise ValueError("Guiding model diagrams support depths zero through three")
    length, width = guide.ell, guide.inst.index_bitsize
    scope = guide.scope_guiding_state
    sort = ("sort_in_place", length, width)
    duplicates = ("duplicates", length, width)
    uncompute = ProbabilisticUncompute(guide.sort_ancilla_bitsize)
    flag = ("multi_control_x", (0, 0))
    x = ("x",)
    root_graph = guide.source_graph(
        root=guide,
        preparation=scope,
        sorting=sort,
        duplicates=duplicates,
        uncompute=uncompute,
        flag_predicate=flag,
        flag_x=x,
    )
    calls: dict[Hashable, dict[Hashable, SymbolicInt]] = {guide: {}}
    for (_, child), count in root_graph.direct_calls.items():
        calls[guide][child] = count
    if max_depth <= 1:
        return symbolic_call_graph(guide, calls, max_depth=max_depth)
    if guide.inst.is_symbolic() or type(length) is not int or type(width) is not int:
        raise ValueError("Deeper guiding model diagrams require concrete instances")
    preparation = scope.state_preparation
    dense, cycles, _ = preparation._components(preparation.target_width)
    calls[scope] = {preparation: 1}
    calls[preparation] = {
        dense: 1,
        ("permutation", preparation.size, cycles, preparation.target_width): 1,
    }
    calls[sort] = source_in_place_sort_calls(length, width)
    comparator = ComparatorDeclaration(width)
    calls[("comparator", width)] = comparator.direct_calls()
    calls[("comparator_adjoint", width)] = comparator.direct_calls(adjoint=True)
    calls[("xor", width)] = {("cx",): width}
    calls[duplicates] = source_unsigned_sorted_duplicates_calls(length, width)
    for adjoint in (False, True):
        calls[("half_less_than", width, adjoint)] = (
            source_unsigned_half_less_than_calls(width, adjoint=adjoint)
        )
    word_width = guide.sort_ancilla_bitsize
    assert isinstance(word_width, int)
    hadamards = ("on_each_h", word_width)
    word_control = ("controlled_x", (0,) * word_width, True)
    calls[uncompute] = {hadamards: 1, word_control: 1}
    calls[hadamards] = {("h",): word_width}
    for pattern in ((0, 0), (1,) * (length - 1)):
        calls[("multi_control_x", pattern)] = {("controlled_x", pattern, False): 1}
    for pattern, word in (((0, 0), False), ((0,) * word_width, True)):
        calls[("controlled_x", pattern, word)] = {
            ("cx",): 1,
            ("ctrl_spec_and", pattern, word, False): 1,
            ("ctrl_spec_and", pattern, word, True): 1,
        }
    return symbolic_call_graph(guide, calls, max_depth=max_depth)


def guiding_call_title(node: Hashable) -> str:
    """Use short domain labels while preserving parameter-bearing graph keys."""
    if not isinstance(node, tuple):
        return type(node).__name__
    name = node[0]
    labels = {
        "sort_in_place": "Sort in place",
        "duplicates": "Duplicate check",
        "comparator": "Comparator",
        "comparator_adjoint": "Comparator adjoint",
        "allocate_bit": "Allocate bit",
        "free_bit": "Free bit",
        "greater_than": "Greater than",
        "greater_than_adjoint": "Greater than adjoint",
        "controlled_swap": "Controlled swap",
        "xor": "XOR",
        "cx": "CNOT",
        "x": "X",
        "h": "H",
        "bitwise_not": "Bitwise NOT",
        "permutation": "Permutation",
    }
    if name == "half_less_than":
        return "Half less than" + (" adjoint" if node[2] else "")
    if name == "out_of_place_add":
        return "Out-of-place add" + (" adjoint" if node[2] else "")
    if name == "on_each_h":
        return f"H on {node[1]} bits"
    if name == "multi_control_x":
        return f"Multi-control X ({len(node[1])} bits)"
    if name == "controlled_x":
        return f"Controlled X ({len(node[1])} bits)"
    if name == "ctrl_spec_and":
        return "Control predicate" + (" adjoint" if node[3] else "")
    return labels.get(name, str(name))
