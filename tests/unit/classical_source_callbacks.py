"""Validate original SDK callback results without changing opaque label objects.

The frozen functions accept duck-typed owners and arbitrary hashable tensor
labels at runtime. Their nominal Bloq/Connection annotations are narrower than
those source-supported calls. Keep that external boundary here; do not change
fixtures into different owners or replace their original callback algorithms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quimb.tensor import Tensor

SourceTruthTable = tuple[
    list[str], list[str], list[tuple[Sequence[object], Sequence[object]]]
]


def source_truth_table(owner: object) -> SourceTruthTable:
    """Call the original function and retain its three original list objects."""
    from qualtran.simulation import classical_sim

    callback: Callable[[object], object] = getattr(
        classical_sim, "get_classical_truth_table"
    )
    assert id(callback) == id(classical_sim.get_classical_truth_table)
    result = callback(owner)
    assert isinstance(result, tuple) and len(result) == 3
    inputs, outputs, rows = result
    assert isinstance(inputs, list) and all(isinstance(v, str) for v in inputs)
    assert isinstance(outputs, list) and all(isinstance(v, str) for v in outputs)
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, tuple) and len(row) == 2
        assert all(isinstance(values, Sequence) for values in row)
    return result


def _tensor_results(result: object) -> list[Tensor]:
    """Retain every original tensor and label after the external callback."""
    from quimb.tensor import Tensor

    assert isinstance(result, list)
    assert all(isinstance(tensor, Tensor) for tensor in result)
    return result


def source_tensors(
    owner: object,
    incoming: Mapping[str, object],
    outgoing: Mapping[str, object],
) -> list[Tensor]:
    """Invoke the exact bound source method with unchanged label mappings."""
    callback: Callable[[Mapping[str, object], Mapping[str, object]], object] = getattr(
        owner, "my_tensors"
    )
    assert callable(callback) and getattr(callback, "__self__") is owner
    return _tensor_results(callback(incoming, outgoing))


def source_classical_action_tensors(
    owner: object,
    incoming: Mapping[str, object],
    outgoing: Mapping[str, object],
) -> list[Tensor]:
    """Invoke the original function on its existing structural source fixture."""
    from qualtran.simulation.tensor import _tensor_from_classical

    callback: Callable[[object, Mapping[str, object], Mapping[str, object]], object] = (
        getattr(_tensor_from_classical, "my_tensors_from_classical_action")
    )
    assert id(callback) == id(_tensor_from_classical.my_tensors_from_classical_action)
    return _tensor_results(callback(owner, incoming, outgoing))
