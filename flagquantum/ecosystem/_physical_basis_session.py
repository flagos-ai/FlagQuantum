# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/interactive_simulator.pybind.cc/.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Convert Python observation objects around persistent physical trajectories."""

import sys
from collections.abc import Callable, Mapping, Sequence, Sized
from fractions import Fraction
from operator import index as integer_index
from secrets import randbits
from typing import SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

from ..compiler.artifacts._array_views import checked_operand_array
from ..compiler.artifacts._operand_views import OperandDomain
from ..compiler.artifacts.operand_array import OperandArray
from ..compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from ..compiler.artifacts.phase_angle import PhaseAngle128
from ..compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from ..compiler.formats.kmb import KmxRegister
from ..compiler.formats.kmx_circuit import KmxCircuit
from ..compiler.registers.layout.operand_word import (
    bind_operand_word,
    is_operand_sequence,
)
from ..compiler.registers.primitives.operand_binding import identify_operand
from ..simulation._basis_trajectories import BasisTrajectories, Wire
from ..simulation.physical_basis_shots import BasisRecord
from ._phase_angle import phase_angle_from_half_turns

_BoolArray = NDArray[np.bool_]
DebugHandler = Callable[
    [OperandIdentity | None, _BoolArray, OperandIdentity | None, _BoolArray], None
]


def _seed_words(seed: object) -> tuple[int, ...]:
    if seed is None:
        return tuple(randbits(32) for _ in range(8))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"Expected seed to be an int or None, but got seed={seed!r}")
    try:
        raw = seed.to_bytes(32, "little")
    except OverflowError as error:
        raise ValueError(f"seed={seed!r} not in range(2**256)") from error
    return tuple(
        int.from_bytes(raw[offset : offset + 4], "little") for offset in range(0, 32, 4)
    )


def _unsigned_index(value: object) -> int:
    if not isinstance(value, SupportsIndex):
        raise TypeError("Shot index must be an integer")
    result = integer_index(value)
    if not 0 <= result < 1 << 64:
        raise ValueError("Shot index must fit an unsigned 64-bit integer")
    return result


def _output(shape: tuple[int, ...], value: object) -> _BoolArray:
    if value is None:
        return np.empty(shape, dtype=np.bool_)
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_:
        raise ValueError("out wasn't a np.ndarray[np.bool_].")
    if value.ndim != len(shape):
        raise ValueError(f"len(out.shape) != {len(shape)}")
    if value.shape != shape:
        raise ValueError(
            f"Expected out.shape == {shape} but got out.shape={value.shape}."
        )
    if not value.flags.writeable:
        raise ValueError("array is not writeable")
    return cast(_BoolArray, value)


def _require_first_cell(array: _BoolArray) -> None:
    # The locked binding obtains a first-cell pointer before its empty loops.
    if array.size == 0:
        raise IndexError("index 0 is out of bounds for an empty observation array")


def _integer_value(value: object, width: int, range_name: str) -> int:
    if type(value) not in (int, bool):
        raise ValueError(f"Expected an int but got new_value={value!r}")
    result = int(cast(int, value))
    if not 0 <= result < 1 << width:
        raise ValueError(f"new_value={value!r} not in range(2**({range_name}={width}))")
    return result


def _input_array(value: _BoolArray, shape: tuple[int, ...]) -> _BoolArray:
    if value.ndim != len(shape):
        raise ValueError(f"len(new_value.shape) != {len(shape)}")
    if value.shape != shape:
        raise ValueError(
            f"Expected new_value.shape == {shape} but got new_value.shape={value.shape}."
        )
    return value


def _wire(item: OperandIdentity) -> Wire:
    kind = {
        OperandKind.QUANTUM: "q",
        OperandKind.CLASSICAL: "b",
        OperandKind.BOOLEAN: "bool",
    }[item.kind]
    return cast(Wire, (kind, item.value))


class PhysicalBasisSession:
    """Resolve native metadata and external array layouts at the owning boundary."""

    def __init__(
        self,
        batch_size: SupportsIndex,
        *,
        debug_print: bool | DebugHandler = True,
        counting_operations: bool = False,
        seed: object = None,
    ) -> None:
        self._state = BasisTrajectories(
            _unsigned_index(batch_size), _seed_words(seed), counting=counting_operations
        )
        self._registers: tuple[KmxRegister, ...] = ()
        self.debug_print = debug_print

    @property
    def batch_size(self) -> int:
        return self._state.shots

    @property
    def counting_operations(self) -> bool:
        return self._state.counting

    @property
    def debug_print(self) -> bool | DebugHandler:
        return self._debug_print

    @debug_print.setter
    def debug_print(self, value: bool | DebugHandler) -> None:
        if not isinstance(value, bool) and not callable(value):
            raise TypeError("Expected debug_print to be a bool or a callable.")
        for position, chunk in enumerate(self._state.chunks):
            chunk.debug_handler = self._debug_handler(position, value)
        self._debug_print = value

    def _debug_handler(
        self, position: int, handler: bool | DebugHandler
    ) -> Callable[[BasisRecord, int, int], None]:
        chunk = self._state.chunks[position]
        count = min(256, self.batch_size - position * 256)

        def observe(record: BasisRecord, value: int, active: int) -> None:
            if isinstance(handler, bool):
                if handler and active & 1:
                    if record.qubits or record.result_bit is not None:
                        sys.stderr.write(str(value & 1))
                    else:
                        sys.stderr.write(
                            f" (phase={PhaseAngle128._from_units(chunk.read_shot_phase(0))})\n"
                        )
                return
            target = None
            if record.qubits:
                target = OperandIdentity(OperandKind.QUANTUM, record.qubits[0])
            elif record.result_bit is not None:
                target = OperandIdentity(OperandKind.CLASSICAL, record.result_bit)
            predicate = (
                None
                if record.condition_bit is None
                else OperandIdentity(OperandKind.CLASSICAL, record.condition_bit)
            )
            handler(
                target,
                np.array(
                    [(value >> lane) & 1 for lane in range(count)], dtype=np.bool_
                ),
                predicate,
                np.array(
                    [(active >> lane) & 1 for lane in range(count)], dtype=np.bool_
                ),
            )

        return observe

    def use_same_registers_as(self, circuit: KmxCircuit) -> None:
        self._registers = circuit.registers

    def do(self, circuit: KmxCircuit) -> None:
        self._state.apply(circuit)

    def clear_for_shot(self) -> None:
        self._state.clear_for_shot()

    def _items(
        self, value: object, *, writing: bool = False
    ) -> tuple[tuple[Wire, ...], bool]:
        register: KmxRegister | None = None
        if isinstance(value, str):
            register = next(
                (item for item in self._registers if item.name == value), None
            )
            if register is None:
                raise IndexError(repr(value))
        elif isinstance(value, OperandIdentity) and value.kind is OperandKind.REGISTER:
            if value.value >= len(self._registers):
                raise IndexError(repr(value))
            register = self._registers[value.value]
        if register is not None:
            return tuple(cast(Wire, member) for member in register.members), False
        singleton = not is_operand_sequence(value)
        try:
            items = OperandArray(
                identify_operand(item) for item in bind_operand_word(value)
            )
        except ValueError as error:
            raise TypeError(
                "Expected a quantum wire, classical bit or basis literal"
            ) from error
        checked_operand_array(items, OperandDomain.Z_OPERAND, "indices")
        if writing and any(item.kind is OperandKind.BOOLEAN for item in items):
            raise TypeError("index contained a Boolean instead of a qubit or bit")
        return tuple(_wire(item) for item in items), singleton

    def read_across_shots(self, indices: object, *, out: object = None) -> object:
        items, singleton = self._items(indices)
        if out is int:
            values = [self._state.read_across(item) for item in items]
            return values[0] if singleton else values
        array = _output(
            (self.batch_size,) if singleton else (len(items), self.batch_size), out
        )
        if singleton or items:
            _require_first_cell(array)
        for row, item in enumerate(items):
            mask = self._state.read_across(item)
            for lane in range(self.batch_size):
                array[(lane,) if singleton else (row, lane)] = bool((mask >> lane) & 1)
        return array

    def read_within_shot(
        self, indices: object, shot_index: object, *, out: object = None
    ) -> object:
        items, singleton = self._items(indices)
        shot = _unsigned_index(shot_index)
        if out is int:
            values = [self._state.read_bit(item, shot) for item in items]
            return (
                values[0]
                if singleton
                else sum(int(bit) << i for i, bit in enumerate(values))
            )
        array = _output(() if singleton else (len(items),), out)
        _require_first_cell(array)
        for offset, item in enumerate(items):
            array[() if singleton else (offset,)] = self._state.read_bit(item, shot)
        return array

    def write_across_shots(self, index: object, new_value: object) -> None:
        items, singleton = self._items(index, writing=True)
        if isinstance(new_value, np.ndarray) and new_value.dtype == np.bool_:
            shape = (self.batch_size,) if singleton else (len(items), self.batch_size)
            array = _input_array(cast(_BoolArray, new_value), shape)
            _require_first_cell(array)
            for item in items:
                self._state.ensure_wire(item)
            for row, item in enumerate(items):
                for lane in range(self.batch_size):
                    self._state.write_bit(
                        item, lane, bool(array[(lane,) if singleton else (row, lane)])
                    )
            return
        if singleton and type(new_value) in (int, bool):
            value = _integer_value(new_value, self.batch_size, "batch_size")
            self._state.write_across(items[0], value)
            return
        if (
            not singleton
            and isinstance(new_value, Sized)
            and hasattr(new_value, "__getitem__")
            and not isinstance(new_value, Mapping)
        ):
            sequence = cast(Sequence[object], new_value)
            if len(sequence) != len(items):
                raise ValueError(
                    f"Expected len(new_value) == len(index) == {len(items)} but got len(new_value)={len(sequence)}."
                )
            for item in items:
                self._state.ensure_wire(item)
            for position, item in enumerate(items):
                item_value = sequence[position]
                self._state.write_across(
                    item, _integer_value(item_value, self.batch_size, "batch_size")
                )
            return
        raise ValueError(f"Don't know how to write from {new_value!r}")

    def write_within_shot(
        self, index: object, shot_index: object, new_value: object
    ) -> None:
        items, _ = self._items(index, writing=True)
        shot = _unsigned_index(shot_index)
        if type(new_value) in (int, bool):
            for item in items:
                self._state.ensure_wire(item)
            value = _integer_value(new_value, len(items), "len(indices)")
            for offset, item in enumerate(items):
                self._state.write_bit(item, shot, bool((value >> offset) & 1))
            return
        if isinstance(new_value, np.ndarray) and new_value.dtype == np.bool_:
            for item in items:
                self._state.ensure_wire(item)
            array = _input_array(cast(_BoolArray, new_value), (len(items),))
            _require_first_cell(array)
            for item, value in zip(items, array, strict=True):
                self._state.write_bit(item, shot, bool(value))
            return
        raise ValueError(f"Don't know how to write from {new_value!r}")

    def read_shot_phase(self, shot_index: int) -> int | Fraction:
        units = self._state.read_phase(_unsigned_index(shot_index))
        if units in (0, 1 << 127):
            return units >> 127
        return Fraction(units, 1 << 127)

    def write_shot_phase(self, shot_index: int, new_phase_half_turns: object) -> None:
        shot = _unsigned_index(shot_index)
        angle = phase_angle_from_half_turns(new_phase_half_turns)
        self._state.write_phase(shot, angle.units)

    def read_phase_flipped_across_shots(self, *, out: object = None) -> object:
        if out is int:
            return self._state.phase_mask()
        array = _output((self.batch_size,), out)
        _require_first_cell(array)
        for shot in range(self.batch_size):
            array[shot] = self._state.phase_flipped(shot)
        return array

    def operation_statistics(
        self, shot_index: object = None
    ) -> PhysicalOperationStatistics:
        if shot_index is not None:
            if not isinstance(shot_index, SupportsIndex):
                raise TypeError("Statistics shot index must be an integer")
            shot_index = integer_index(shot_index)
            if not -(1 << 63) <= shot_index < 1 << 63:
                raise TypeError(
                    "Statistics shot index must fit a signed 64-bit integer"
                )
        return PhysicalOperationStatistics.from_counts(
            *self._state.counter_snapshot(shot_index)
        )

    def clear_operation_statistics(self) -> None:
        self._state.clear_operation_statistics()
