# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix sim/sim.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Own persistent physical numerical state over the shared packed executor."""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from typing import Literal, Protocol

from ..core._fixed_width_integer import FixedWidthInteger
from ._packed_chacha import PackedChaCha
from ._packed_word import pack_lanes
from ._register_buffer import _Register, pack_register_buffer, unpack_register_buffer
from .physical_basis_shots import (
    BasisRecord,
    PackedBasisState,
    apply_packed_basis_records,
)
from .physical_operation_counters import PhysicalOperationCounters


class _Circuit(Protocol):
    @property
    def registers(self) -> Sequence[_Register]: ...
    @property
    def qubits(self) -> int: ...
    @property
    def bits(self) -> int: ...
    @property
    def records(self) -> Sequence[BasisRecord]: ...


class PackedBasisSimulation:
    """Compose state, counters, random draws and buffers with explicit ownership."""

    def __init__(
        self, width: int, draws: Iterator[int], *, counting: bool = False
    ) -> None:
        if width not in (64, 128, 256, 512):
            raise ValueError("Packed simulation width must be 64, 128, 256 or 512")
        self.width = width
        self.rng = draws
        self.random_state = PackedChaCha(width).randomize(
            pack_lanes((next(draws) for _ in range(width // 64)), 64) for _ in range(11)
        )
        self.state = PackedBasisState(defaultdict(int), defaultdict(int), [0] * width)
        self.qubits = 0
        self.bits = 0
        self.registers: tuple[_Register, ...] = ()
        self.register_buffer: list[list[FixedWidthInteger]] = [[] for _ in range(width)]
        self.expected_buffer: list[list[FixedWidthInteger]] = [[] for _ in range(width)]
        self.counters = PhysicalOperationCounters(width, enabled=counting)
        self.debug_handler: Callable[[BasisRecord, int, int], None] | None = None

    def __copy__(self) -> "PackedBasisSimulation":
        return deepcopy(self)

    def configure_for(self, circuit: _Circuit) -> None:
        """Reset circuit state and buffers, retaining random streams and counters."""
        self.registers = tuple(circuit.registers)
        self.register_buffer = self.new_register_buffer()
        self.expected_buffer = self.new_register_buffer()
        self.qubits, self.bits = circuit.qubits, circuit.bits
        self.clear_for_shot()

    def new_register_buffer(self) -> list[list[FixedWidthInteger]]:
        return [
            [FixedWidthInteger(len(register.members)) for register in self.registers]
            for _ in range(self.width)
        ]

    def clear_for_shot(self) -> None:
        """Clear wires, phases and conditions while preserving scalar buffers."""
        self.state.qubits.clear()
        self.state.bits.clear()
        self.state.phases[:] = [0] * self.width
        self.state.sign = 0
        self.state.conditions.clear()

    def ensure_big_enough_state_for(self, qubits: int, bits: int) -> None:
        """Grow dimensions independently with source doubling and retained values."""
        if any(type(value) is not int or value < 0 for value in (qubits, bits)):
            raise ValueError("State dimensions must be nonnegative integers")
        if qubits > self.qubits:
            self.qubits = max(self.qubits * 2, qubits)
        if bits > self.bits:
            self.bits = max(self.bits * 2, bits)

    def safe_read(self, kind: Literal["q", "b", "bool"], index: int) -> int:
        """Read an existing wire or a source scalar Boolean without allocation."""
        if kind == "bool":
            return ((1 << self.width) - 1) * bool(index)
        values, size = (
            (self.state.qubits, self.qubits)
            if kind == "q"
            else (self.state.bits, self.bits)
        )
        return values.get(index, 0) if 0 <= index < size else 0

    def read_shot_phase(self, lane: int) -> int:
        if not 0 <= lane < self.width:
            raise IndexError("Phase lane is outside the packed width")
        return (
            self.state.phases[lane] + ((self.state.sign >> lane) & 1) * (1 << 127)
        ) % (1 << 128)

    def copy_register_buffer_into_state(self, register: int | None = None) -> None:
        if register is None:
            pack_register_buffer(self.registers, self.register_buffer, self.state)
        else:
            pack_register_buffer(
                (self.registers[register],),
                [[row[register]] for row in self.register_buffer],
                self.state,
            )

    def copy_state_into_register_buffer(self, register: int | None = None) -> None:
        if register is None:
            unpack_register_buffer(self.registers, self.register_buffer, self.state)
        else:
            column = [[row[register]] for row in self.register_buffer]
            unpack_register_buffer((self.registers[register],), column, self.state)
            for row, value in zip(self.register_buffer, column, strict=True):
                row[register] = value[0]

    def next_measurement(self) -> int:
        word, self.random_state = self.random_state.next_word()
        return word

    def apply(self, circuit: _Circuit, *, counts: Counter[str] | None = None) -> None:
        def observe(record: BasisRecord, active: int) -> None:
            if (
                record.opcode.startswith("DEBUG_PRINT")
                and self.debug_handler is not None
            ):
                if record.qubits:
                    value = self.state.qubits[record.qubits[0]]
                elif record.result_bit is not None:
                    value = self.state.bits[record.result_bit]
                else:
                    value = 0
                self.debug_handler(record, value, active)
            self.counters.observe(record, active)
            if counts is not None:
                counts[record.opcode] += active.bit_count()

        apply_packed_basis_records(
            circuit.records, self.state, self.next_measurement, observer=observe
        )

    def clear_operation_statistics(self) -> None:
        self.counters.clear()
