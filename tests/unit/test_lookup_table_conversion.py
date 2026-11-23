"""Validate complete table utility conversion through existing lookup owners."""

import json
from collections import Counter
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any, Literal, SupportsIndex, overload

import pytest
import torch

from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.registers.lookup.lookup_binding import lookup_binding
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession
from tests.unit.circuit_asset_fixtures import instructions
from tests.unit.fixtures.generate_lookup_table_conversion import conversion_table
from tests.unit.measurement_checks import expand_reset_branches, forced_branch
from tests.unit.test_lookup_binding import native_trace
from tests.unit.test_physical_circuit_binding import observe

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
    target = tuple(range(n, n + row["width"]))
    workspace = tuple(range(n + 1, 2 * n + 1))
    if row["error"] is None:
        actual = lookup_binding(table, address, target, workspace=workspace)
        text = "\n".join(encode_kmx_block_marks(*actual).splitlines()[1:])
        assert text == row["text"]
        assert observe(read_kmx_circuit(text)) == row["execution"]
        assert events == row["events"]
        events.clear()
        erase = lookup_binding(
            table,
            address,
            target,
            workspace=workspace,
            erase=True,
            reused_phase_bits=(0,) if n and target else (),
        )
        joined = "\n".join(
            part
            for part in (
                text,
                "\n".join(encode_kmx_block_marks(*erase).splitlines()[1:]),
            )
            if part
        )
        assert observe(read_kmx_circuit(joined)) == row["joined_execution"]
        assert events == row["joined_events"]
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


def test_malformed_callback_storage_refuses_before_native_encoding() -> None:
    class ShortBytes(int):
        def to_bytes(
            self,
            length: SupportsIndex = 1,
            byteorder: Literal["little", "big"] = "big",
            *,
            signed: bool = False,
        ) -> bytes:
            return b""

    class InvalidLength(int):
        def bit_length(self) -> Any:
            return "invalid"

    class OverflowLength(int):
        def bit_length(self) -> int:
            return 1 << 64

    # The source would read beyond the returned short buffer. Do not execute it.
    with pytest.raises(ValueError, match="enough bytes"):
        lookup_binding([ShortBytes(1)], (), (0,))
    for value in (InvalidLength(1), OverflowLength(1)):
        with pytest.raises(TypeError, match="storage size"):
            lookup_binding([value], (), (0,))


@pytest.mark.parametrize(
    "row",
    [
        row
        for row in _DATA["conversion_observations"]
        if row["error"] is None and row["width"] == 1 and len(row["events"]) >= 3
    ],
)
def test_subclass_joined_every_complex_branch_restores_clean_work(
    row: dict[str, Any],
) -> None:
    events: list[str] = []
    table = conversion_table(row["label"], events)
    parts = [
        "\n".join(
            encode_kmx_block_marks(
                *lookup_binding(table, (), (0,), erase=erase)
            ).splitlines()[1:]
        )
        for erase in (False, True)
    ]
    joined = read_kmx_circuit("\n".join(parts))
    assert joined.to_kmx() == row["joined_execution"]["text"]
    physical = type(joined)(
        (), tuple(record for record in joined if record.tag is None)
    )
    gates = expand_reset_branches(instructions(physical), first_slot=10)
    initial = torch.tensor([0, 0, 1 + 2j, 3 - 4j], dtype=torch.complex128)
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    expected[:2] = initial[2:]
    count = sum(gate.name == "measure" for gate in gates)
    total = torch.zeros_like(initial.real)
    for outcomes in product((0, 1), repeat=count):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual,
            expected * (-1) ** outcomes[0] / 2 ** (count / 2),
            rtol=0,
            atol=2e-15,
        )
        assert torch.count_nonzero(actual[2:]) == 0
        total += actual.abs().square()
    torch.testing.assert_close(total, expected.abs().square(), rtol=0, atol=2e-15)
