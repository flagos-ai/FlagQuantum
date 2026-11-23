"""Validate complete table utility conversion through existing lookup owners."""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

import pytest

from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.registers.lookup.lookup_binding import lookup_binding
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession
from tests.unit.fixtures.generate_lookup_table_conversion import conversion_table
from tests.unit.test_lookup_binding import native_trace

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "lookup_table_conversion.json").read_text()
)


@pytest.mark.parametrize("row", _DATA["rows"])
def test_cross_byte_bit_order_complete_callers_and_actual_counts(
    row: dict[str, Any],
) -> None:
    case = row["input"]
    program, calls = native_trace(case)
    text = "\n".join(encode_kmx_block_marks(program, calls).splitlines()[1:])
    assert text == row["text"]
    assert (
        dict(Counter(line.split()[0] for line in text.splitlines()))
        == row["static_counts"]
    )
    session = PhysicalBasisSession(
        3, seed=0, debug_print=False, counting_operations=True
    )
    n, width = case["address_width"], case["target_width"]
    for wire in range(n + width):
        session.write_across_shots(wire, (wire * 5 + 3) & 7)
    session.do(read_kmx_circuit(text))
    assert [
        session.read_across_shots(wire, out=int) for wire in range(2 * n + width)
    ] == row["values"]
    assert [str(session.read_shot_phase(lane)) for lane in range(3)] == row["phases"]
    assert dict(session.operation_statistics()) == row["executed_counts"]
    assert all(value == 0 for value in row["values"][n + width :])
    if case["route"] == "joined":
        assert all(value == 0 for value in row["values"][n:])


@pytest.mark.parametrize("row", _DATA["conversion_observations"])
def test_conversion_failure_order_and_explicit_subclass_boundary(
    row: dict[str, Any],
) -> None:
    events: list[str] = []
    table = conversion_table(row["label"], events)
    n = 0 if len(table) == 1 else 1
    address = tuple(range(n))
    target = (n,)
    workspace = tuple(range(n + 1, 2 * n + 1))
    if row["label"] == "subclass-boundary":
        # Native int normalization is an existing, separately open source difference.
        assert row["error"] == "ValueError" and row["events"] == ["bit_length"]
        actual = lookup_binding(table, address, target, workspace=workspace)
        assert actual == lookup_binding([1], address, target, workspace=workspace)
        assert events == []
        return
    if row["error"] is None:
        actual = lookup_binding(table, address, target, workspace=workspace)
        assert actual == lookup_binding([0, 1], address, target, workspace=workspace)
    else:
        expected = ValueError if row["error"] == "ValueError" else RuntimeError
        with pytest.raises(expected) as caught:
            lookup_binding(table, address, target, workspace=workspace)
        assert str(caught.value) == row["message"]
    assert events == row["events"]


def test_indexed_table_length_is_read_once_and_failures_stop_access() -> None:
    events: list[str | int] = []

    class Table(Sequence[object]):
        def __len__(self) -> int:
            events.append("len")
            return 2

        @overload
        def __getitem__(self, index: int) -> object: ...

        @overload
        def __getitem__(self, index: slice) -> Sequence[object]: ...

        def __getitem__(self, index: int | slice) -> object:
            assert isinstance(index, int)
            events.append(index)
            return index

    assert lookup_binding(Table(), (0,), (1,), workspace=(2,)) == lookup_binding(
        [0, 1], (0,), (1,), workspace=(2,)
    )
    assert events == ["len", 0, 1]


def test_zero_width_still_converts_and_validates_every_row() -> None:
    program, _ = lookup_binding(["0", -0.9, False], (), ())
    assert program.operations == ()
    for index in (0, 1, 2):
        table = [0, 0, 0]
        table[index] = 1
        with pytest.raises(ValueError, match=rf"table\[{index}\]=1"):
            lookup_binding(table, (), ())
