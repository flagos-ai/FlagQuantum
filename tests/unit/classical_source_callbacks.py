"""Validate original SDK callback results without changing opaque label objects.

The frozen functions accept duck-typed owners and arbitrary hashable tensor
labels at runtime. Their nominal Bloq/Connection annotations are narrower than
those source-supported calls. Keep that external boundary here; do not change
fixtures into different owners or replace their original callback algorithms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from qualtran._infra.quantum_graph import Soquet
    from qualtran._infra.registers import Register
    from qualtran.simulation.tensor._quimb import DiscardInd

    from tests._reference_dependencies import unavailable_reference

    try:
        import numpy as np
        from numpy.typing import NDArray
    except ModuleNotFoundError as error:
        unavailable_reference(error)
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


def source_measurement_tensors(
    owner: object,
    incoming: Mapping[str, object],
    outgoing: Mapping[str, object],
) -> tuple[Tensor, DiscardInd]:
    """Retain the original destructive instrument and its trace marker."""
    from qualtran.simulation.tensor._quimb import DiscardInd
    from quimb.tensor import Tensor

    callback: Callable[[Mapping[str, object], Mapping[str, object]], object] = getattr(
        owner, "my_tensors"
    )
    assert callable(callback) and getattr(callback, "__self__") is owner
    result = callback(incoming, outgoing)
    assert isinstance(result, list) and len(result) == 2
    tensor, discarded = result
    assert isinstance(tensor, Tensor) and isinstance(discarded, DiscardInd)
    return tensor, discarded


def source_discard_tensors(
    owner: object,
    incoming: Mapping[str, object],
    outgoing: Mapping[str, object],
) -> list[DiscardInd]:
    """Keep the original trace-marker list with its opaque source labels."""
    from qualtran.simulation.tensor._quimb import DiscardInd

    callback: Callable[[Mapping[str, object], Mapping[str, object]], object] = getattr(
        owner, "my_tensors"
    )
    assert callable(callback) and getattr(callback, "__self__") is owner
    result = callback(incoming, outgoing)
    assert isinstance(result, list) and all(isinstance(v, DiscardInd) for v in result)
    return result


def source_discard_index(label: object, bit: int) -> DiscardInd:
    """Construct the exact original marker for a source-supported opaque label."""
    from qualtran.simulation.tensor import _quimb

    factory: Callable[[tuple[object, int]], object] = getattr(_quimb, "DiscardInd")
    result = factory((label, bit))
    assert isinstance(result, _quimb.DiscardInd)
    assert result.ind_tuple[0] is label and result.ind_tuple[1] == bit
    return result


def source_controlled_tensor(owner: object) -> NDArray[np.float64 | np.complex128]:
    """Read the unchanged source controlled tensor without replacing its owner."""
    from tests._reference_dependencies import unavailable_reference

    try:
        import numpy as np
    except ModuleNotFoundError as error:
        unavailable_reference(error)
    callback: Callable[[], object] = getattr(owner, "_tensor_data")
    assert callable(callback) and getattr(callback, "__self__") is owner
    result = callback()
    assert isinstance(result, np.ndarray)
    assert result.dtype in (np.dtype("float64"), np.dtype("complex128"))
    return result


def source_placeholder_soquet(register: Register, index: tuple[int, ...]) -> Soquet:
    """Retain the original synthetic None owner used only as a tensor label."""
    from qualtran._infra import quantum_graph

    factory: Callable[[None, Register, tuple[int, ...]], object] = getattr(
        quantum_graph, "Soquet"
    )
    result = factory(None, register, index)
    assert isinstance(result, quantum_graph.Soquet) and result.binst is None
    return result


class _ClassicalValueCallback(Protocol):
    def __call__(self, **values: object) -> object: ...


def source_classical_values(owner: object, **values: object) -> dict[str, object]:
    """Keep actual source value domains wider than their nominal array annotations."""
    callback: _ClassicalValueCallback = getattr(owner, "on_classical_vals")
    assert callable(callback) and getattr(callback, "__self__") is owner
    result = callback(**values)
    assert isinstance(result, dict) and all(isinstance(key, str) for key in result)
    return result
