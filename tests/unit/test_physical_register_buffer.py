"""Exercise mixed-register packing through the shared numerical executor."""

from collections import Counter, defaultdict

import pytest

from flagquantum.compiler.formats.kmb import KmxRegister
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.core._fixed_width_integer import FixedWidthInteger
from flagquantum.simulation._register_buffer import (
    pack_register_buffer,
    unpack_register_buffer,
)
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", (0, 1, 63, 64, 65, 129))
@pytest.mark.parametrize("lanes", (1, 3, 64, 128))
def test_register_transposition_mixed_widths_and_lanes(width: int, lanes: int) -> None:
    registers = (
        KmxRegister("mixed", tuple(("q" if i % 2 else "b", i) for i in range(width))),
    )
    rows = [
        [
            FixedWidthInteger(
                width,
                ((lane + 1) * 0xFEDCBA9876543210FEDCBA987654321) & ((1 << width) - 1),
            )
        ]
        for lane in range(lanes)
    ]
    original = [row.copy() for row in rows]
    state = PackedBasisState(defaultdict(int), defaultdict(int), [0] * lanes)
    pack_register_buffer(registers, rows, state)
    for offset, (kind, wire) in enumerate(registers[0].members):
        source = state.qubits if kind == "q" else state.bits
        assert source[wire] == sum(
            ((row[0].value >> offset) & 1) << lane for lane, row in enumerate(original)
        )
    unpack_register_buffer(registers, rows, state)
    assert rows == original


def test_overlapping_members_last_write_and_hanging_bits() -> None:
    registers = (
        KmxRegister("first", (("q", 0), ("b", 0))),
        KmxRegister("last", (("q", 0),)),
    )
    rows = [[FixedWidthInteger(2, 3), FixedWidthInteger(1, 2)] for _ in range(64)]
    state = PackedBasisState(defaultdict(int), defaultdict(int), [0] * 64)
    pack_register_buffer(registers, rows, state)
    assert state.qubits[0] == 0 and state.bits[0] == (1 << 64) - 1
    counts: Counter[str] = Counter()
    apply_packed_basis_records(
        read_kmx_circuit("X q0 if b0").records, state, lambda: 0, counts=counts
    )
    unpack_register_buffer(registers, rows, state)
    assert all(
        row == [FixedWidthInteger(2, 3), FixedWidthInteger(1, 1)] for row in rows
    )
    assert counts == {"X": 64}


def test_invalid_register_buffer_fails_before_assignment() -> None:
    registers = (KmxRegister("x", (("q", 0),)),)
    state = PackedBasisState(defaultdict(int), defaultdict(int), [0])
    cases: tuple[tuple[list[list[FixedWidthInteger]], str], ...] = (
        ([], "lane"),
        ([[]], "layout"),
        ([[FixedWidthInteger(2)]], "layout"),
    )
    for rows, message in cases:
        with pytest.raises(ValueError, match=message):
            pack_register_buffer(registers, rows, state)
        with pytest.raises(ValueError, match=message):
            unpack_register_buffer(registers, rows, state)
    assert not state.qubits and not state.bits
