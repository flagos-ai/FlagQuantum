"""Convert concrete external sparse oracles to owned native snapshots.

The optional dependency ends here. Independent logical models are supplied by
the caller; emitted primitive gates never manufacture a source cost model.
"""

from typing import Any, cast

from ...algorithms.primitives.circuit_sparse_oracles import (
    CircuitColumnOracle,
    CircuitEntryOracle,
    CircuitSqrtEntryOracle,
)
from ...compiler._component_trace.resource_model import ComponentCost
from ._component_snapshot import _snapshot


def _width(source: Any) -> int:
    value = source.system_bitsize
    if type(value) is not int or value < 1:
        raise ValueError(
            "Foreign oracle binding requires a positive concrete index width"
        )
    return value


def import_column_oracle(
    source: object, *, models: tuple[ComponentCost, ComponentCost]
) -> CircuitColumnOracle:
    """Import a concrete column object's actual forward and adjoint operations."""
    foreign = cast(Any, source)
    width = _width(foreign)
    ports = (("l", width), ("i", width))
    forward, first = _snapshot(foreign, ports)
    backward, second = _snapshot(foreign.adjoint(), ports)
    return CircuitColumnOracle(
        width, foreign.num_nonzero, forward, backward, models, (first, second)
    )


def import_sqrt_entry_oracle(
    source: object, *, models: tuple[ComponentCost, ComponentCost]
) -> CircuitSqrtEntryOracle:
    """Import a concrete entry, retaining independent models and precision."""
    foreign = cast(Any, source)
    width = _width(foreign)
    ports = (("q", 1), ("i", width), ("j", width))
    forward, first = _snapshot(foreign, ports)
    backward, second = _snapshot(foreign.adjoint(), ports)
    return CircuitSqrtEntryOracle(
        width, forward, backward, models, foreign.epsilon, (first, second)
    )


def import_entry_oracle(
    source: object, *, models: tuple[ComponentCost, ComponentCost]
) -> CircuitEntryOracle:
    """Import actual amplitude oracles without requiring a sqrt precision port.

    The source's forward and adjoint define the coefficient convention; no
    square root, requantization or synthesized inverse is substituted here.
    """
    foreign = cast(Any, source)
    width = _width(foreign)
    ports = (("q", 1), ("i", width), ("j", width))
    forward, first = _snapshot(foreign, ports)
    backward, second = _snapshot(foreign.adjoint(), ports)
    return CircuitEntryOracle(
        width, forward, backward, models, binding_swaps=(first, second)
    )
