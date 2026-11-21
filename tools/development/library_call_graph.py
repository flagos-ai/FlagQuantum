# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/all_call_graph.py.

"""Build a library-wide declared graph from existing lazy component examples."""

import logging
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, TypeVar

from flagquantum.compiler._component_trace.call_graph import (
    CallDeclaration,
    Generalizer,
    symbolic_call_graph_union,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.testing.component_examples import ComponentExample

if TYPE_CHECKING:
    from flagquantum.compiler._component_trace.resource_generalization import (
        ResourceCall,
    )

Node = TypeVar("Node", bound=Hashable)
logger = logging.getLogger(__name__)


def library_call_graph(
    examples: Iterable[ComponentExample[Node]],
    declared_calls: CallDeclaration[Node],
    *,
    generalizer: Generalizer[Node] | Sequence[Generalizer[Node]] | None = None,
    keep: Callable[[Node], bool] | None = None,
) -> dict[Node, dict[Node, SymbolicInt]]:
    """Union example graphs using the Compiler's declaration traversal.

    The owning domain supplies its existing call declarations, generalizers
    and terminal predicate. Example factories are evaluated once in sequence.
    The result is weighted adjacency, including isolated roots and leaves.
    This report describes declared models rather than emitted primitive gates.
    """

    def components() -> Iterable[Node]:
        for example in examples:
            logger.info("Building call graph for: %s", example.name)
            yield example.make()

    return symbolic_call_graph_union(
        components(), declared_calls, generalizer=generalizer, keep=keep
    )


def resource_library_call_graph(
    examples: Iterable[ComponentExample["ResourceCall"]],
    declared_calls: CallDeclaration["ResourceCall"],
) -> dict["ResourceCall", dict["ResourceCall", SymbolicInt]]:
    """Apply the source library profile using existing native resource owners.

    The domain supplies direct model declarations. Missing declared capabilities
    are retained as warned terminal nodes. Explicit T, Toffoli and rotation keys
    are terminal independently of the supplied declaration callback.
    """
    import warnings
    from collections.abc import Mapping

    from flagquantum.compiler._component_trace.call_graph import SymbolAllocator
    from flagquantum.compiler._component_trace.resource_classification import (
        ignore_known_cliffords,
    )
    from flagquantum.compiler._component_trace.resource_generalization import (
        FixedTCall,
        PhaseCall,
        RotationCall,
        generalize_control_values,
        generalize_rotation,
        ignore_allocation,
        ignore_register_partition,
    )
    from flagquantum.errors import CapabilityError

    def keep(node: "ResourceCall") -> bool:
        if node in (FixedTCall("toffoli"), PhaseCall("t")) or isinstance(
            node, RotationCall
        ):
            return True
        try:
            raw = declared_calls(node, SymbolAllocator())
            if not isinstance(raw, Mapping):
                tuple(raw)
        except CapabilityError:
            warnings.warn(f"{node} lacks a call graph.", stacklevel=2)
            return True
        return False

    return library_call_graph(
        examples,
        declared_calls,
        generalizer=[
            ignore_register_partition,
            generalize_control_values,
            generalize_rotation,
            ignore_allocation,
            ignore_known_cliffords,
        ],
        keep=keep,
    )
