"""Compare complete physical statistics with the unchanged frozen binding."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from flagquantum.compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from tests.unit.fixtures.generate_operation_statistics import snapshot

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "operation_statistics.json").read_text()
)


@pytest.mark.parametrize("record", _DATA["static"])
def test_complete_static_source_projection(record: list[object]) -> None:
    text, expected = record
    assert isinstance(text, str)
    counts = PhysicalOperationStatistics.from_circuit(read_kmx_circuit(text))
    assert snapshot(counts) == expected
    assert dict(counts) == dict(counts.items())
    assert list(counts) == counts.keys()
    assert counts == PhysicalOperationStatistics.from_circuit(read_kmx_circuit(text))
    assert counts != object()
    copied = counts.z_pow_clifford_hierarchy_counts
    copied.clear()
    assert snapshot(counts) == expected


@pytest.mark.parametrize("row", _DATA["ranges"])
def test_full_source_rotation_ranges(row: list[int | None]) -> None:
    text = _DATA["static"][5][0]
    counts = PhysicalOperationStatistics.from_circuit(read_kmx_circuit(text))
    low, high, expected = row
    assert low is not None
    assert counts.num_z_pow(low, high) == expected


def test_exact_mapping_equality_and_owned_snapshots() -> None:
    empty = PhysicalOperationStatistics()
    assert not empty
    assert empty["X"] == 0
    marker = object()
    assert empty.get("X", marker) is marker
    assert empty.get("UNKNOWN", marker) is marker
    assert empty.get("UNKNOWN") is None
    assert "X" not in empty and b"X" not in empty and 123 not in empty
    with pytest.raises(KeyError):
        _ = empty["UNKNOWN"]
    left = PhysicalOperationStatistics.from_circuit(read_kmx_circuit("X q0"))
    right = PhysicalOperationStatistics.from_circuit(read_kmx_circuit("X q0 if b0"))
    assert dict(left) == dict(right) and left != right
    assert "X" in left and b"X" not in left
    assert left[b"X"] == left[bytearray(b"X")] == left.get(b"X") == 1
    assert left.get(bytearray(b"X")) == 1
    assert left.get("CX", 7) == 7
    assert left.get("X", marker) == 1
    quarter = PhysicalOperationStatistics.from_circuit(
        read_kmx_circuit("Z_POW q0 0.25")
    )
    eighth = PhysicalOperationStatistics.from_circuit(
        read_kmx_circuit("Z_POW q0 0.125")
    )
    assert dict(quarter) == dict(eighth) and quarter != eighth
    keys = left.keys()
    keys.clear()
    assert left.keys() == ["X"]
    with pytest.raises(FrozenInstanceError):
        setattr(left, "raw_op_counts", ())
    with pytest.raises(TypeError, match="key"):
        left.__getitem__(None)  # type: ignore[index]
    with pytest.raises(UnicodeDecodeError):
        _ = left[b"\xff"]
    with pytest.raises(TypeError, match="signed 64-bit"):
        left.num_z_pow(1 << 63)


def test_counter_capacity_and_uint64_wrap() -> None:
    maximum = (1 << 64) - 1
    counts = PhysicalOperationStatistics.from_counts(
        (("X", maximum), ("X", 1), ("CCX", maximum), ("CCX_IF", 1)),
        ((3, maximum), (3, 1), (128, 2)),
    )
    assert counts["X"] == counts.num_toffoli_like == counts.num_t_like == 0
    assert counts.num_instructions == 0
    assert counts.num_non_clifford == counts.num_z_pow() == 2
    assert counts.z_pow_clifford_hierarchy_counts == {128: 2}
    with pytest.raises(TypeError, match="immutable"):
        PhysicalOperationStatistics([0] * 256)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="256"):
        PhysicalOperationStatistics((), ())
    raw = [0] * 256
    raw[0] = -1
    with pytest.raises(ValueError, match="unsigned"):
        PhysicalOperationStatistics(tuple(raw))
    for invalid in (-1, 1 << 64, True):
        with pytest.raises(ValueError, match="unsigned"):
            PhysicalOperationStatistics.from_counts((("X", invalid),), ())
        with pytest.raises(ValueError, match="unsigned"):
            PhysicalOperationStatistics.from_counts((), ((3, invalid),))
    for invalid in (-1, 129, True):
        with pytest.raises(ValueError, match="hierarchy"):
            PhysicalOperationStatistics.from_counts((), ((invalid, 1),))
    with pytest.raises(KeyError):
        PhysicalOperationStatistics.from_counts((("UNKNOWN", 1),), ())
