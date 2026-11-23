# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix MutableCircuit.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/mutable_circuit.cc
# Register lifetime and mark contexts also follow py/circuit/circuit_builder.pybind.cc.

"""Build raw reversible circuits incrementally and freeze a validated value."""

from collections.abc import Sequence
from copy import deepcopy
from itertools import chain
from math import inf
from types import TracebackType
from typing import BinaryIO, Literal

from ..artifacts._operand_tags import MAX_NUM_QUBITS as _MAX_WIRES
from ..artifacts._operand_tags import MAX_NUM_REGISTERS as _MAX_REGISTERS
from ..artifacts._unsigned_size import unsigned_size
from ..artifacts.operand_array import OperandArray
from ..artifacts.operand_identity import OperandIdentity, OperandKind
from ..registers.layout.operand_word import bind_operand_word
from ..registers.layout.wire_allocation import WireAllocation
from ..registers.primitives.operand_binding import identify_operand
from .kmb import (
    _REGISTER_QUANTUM,
    KmxRegister,
    _append_kmx_text,
    _append_operation,
    _Columns,
    _pack_kmb,
    _RegisterData,
)
from .kmb_operations import INFO_TAG, OPERATION_CODES
from .kmx_circuit import KmxCircuit, _freeze_columns


class KmxCircuitBuilder:
    """Own mutable operation columns with an immutable circuit output boundary."""

    def __init__(self) -> None:
        self._columns = _Columns()
        self._quantum_storage = WireAllocation()
        self._classical_storage = WireAllocation()
        self._max_qubits: int | float = inf

    def __copy__(self) -> "KmxCircuitBuilder":
        duplicate = KmxCircuitBuilder()
        duplicate._quantum_storage = deepcopy(self._quantum_storage)
        duplicate._classical_storage = deepcopy(self._classical_storage)
        duplicate._max_qubits = self._max_qubits
        duplicate._columns.registers = [
            _RegisterData(register.name, register.members.copy())
            for register in self._columns.registers
        ]
        duplicate.append(self)
        return duplicate

    @property
    def max_qubits(self) -> int | float:
        return self._max_qubits

    @property
    def num_allocated_qubits(self) -> int:
        return self._quantum_storage.allocated_count

    @property
    def num_free_qubits(self) -> int | float:
        if self._max_qubits == inf:
            return inf
        remaining = int(self._max_qubits) - self.num_allocated_qubits
        return remaining - (1 << 64) if remaining >= 1 << 63 else remaining

    def set_max_qubits(self, limit: int | float) -> None:
        """Set the live quantum budget while retaining already allocated wires."""
        if isinstance(limit, float) and limit >= float.fromhex("0x1.ffffffp+127"):
            limit = inf
        elif not isinstance(limit, int) or not 0 <= limit < 1 << 64:
            raise ValueError(
                "Quantum budget requires a nonnegative integer or infinity"
            )
        else:
            limit = int.__int__(limit)
        if limit < self.num_allocated_qubits:
            raise ValueError("Quantum budget is smaller than the live allocation")
        self._max_qubits = inf if limit == (1 << 64) - 1 else limit

    def alloc_qubits(self, count: int) -> OperandArray:
        """Allocate ordered temporary wires without resetting their physical state."""
        count = self._check_quantum_budget(count)
        return OperandArray(
            OperandIdentity(OperandKind.QUANTUM, wire)
            for wire in self._quantum_storage.allocate(count)
        )

    def alloc_bits(self, count: int) -> OperandArray:
        """Allocate classical temporary IDs independently of the quantum budget."""
        return OperandArray(
            OperandIdentity(OperandKind.CLASSICAL, wire)
            for wire in self._classical_storage.allocate(count)
        )

    def free(self, arg: object) -> None:
        """Release converted temporary operands sequentially, preserving LIFO reuse."""
        operands = tuple(identify_operand(value) for value in bind_operand_word(arg))
        for operand in operands:
            if operand.kind is OperandKind.QUANTUM:
                self._quantum_storage.free(operand.value)
            elif operand.kind is OperandKind.CLASSICAL:
                self._classical_storage.free(operand.value)
            else:
                raise ValueError("Physical release requires quantum or classical IDs")

    def create_quantum_register(self, length: int, name: str) -> OperandArray:
        """Reserve permanent quantum IDs and append their named membership."""
        self._check_register_name(name)
        length = self._check_quantum_budget(length)
        wires = self._quantum_storage.reserve(length)
        return self._append_register(wires, name, OperandKind.QUANTUM)

    def create_classical_register(self, length: int, name: str) -> OperandArray:
        """Reserve permanent classical IDs independently of temporary ownership."""
        self._check_register_name(name)
        wires = self._classical_storage.reserve(length)
        return self._append_register(wires, name, OperandKind.CLASSICAL)

    def _check_register_name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("A physical register name must be a string")
        if len(self._columns.registers) >= _MAX_REGISTERS:
            raise ValueError("KMX register count exceeds the format limit")

    def _check_quantum_budget(self, count: int) -> int:
        count = unsigned_size(count)
        if count + self.num_allocated_qubits > self._max_qubits:
            raise ValueError("Quantum allocation exceeds the live quantum budget")
        return count

    def _append_register(
        self, wires: tuple[int, ...], name: str, kind: OperandKind
    ) -> OperandArray:
        quantum = _REGISTER_QUANTUM if kind is OperandKind.QUANTUM else 0
        self._columns.registers.append(
            _RegisterData(name, [wire | quantum for wire in wires])
        )
        return OperandArray(OperandIdentity(kind, wire) for wire in wires)

    def mark(self, name: str) -> "_BuilderMark":
        """Create a reusable block context that propagates body exceptions."""
        if not isinstance(name, str):
            raise TypeError("A physical block name must be a string")
        return _BuilderMark(self, name)

    @property
    def operation_count(self) -> int:
        """Count parsed physical records without freezing."""
        return len(self._columns.operations)

    @property
    def qubits(self) -> int:
        """Inspect the inferred width even before alias validation."""
        members = (
            value & ~_REGISTER_QUANTUM
            for register in self._columns.registers
            for value in register.members
            if value & _REGISTER_QUANTUM
        )
        operands = chain.from_iterable(self._columns.words[:8])
        return max(chain(members, operands), default=-1) + 1

    def append_kmx_line(self, text: str) -> None:
        """Parse one file-style line, preserving hashes in quoted metadata."""
        if "\r" in text or "\n" in text:
            raise ValueError("Expected one KMX line")
        _append_kmx_text(self._columns, text, file_text=True)

    def append_kmx_text(self, text: str) -> None:
        """Append KMX statements, retaining complete statements before an error."""
        _append_kmx_text(self._columns, text, file_text=False)

    def append_kmx_file(self, source: BinaryIO) -> None:
        """Append file input from the caller's current binary stream position."""
        for line in source:
            _append_kmx_text(self._columns, line.decode(), file_text=True)

    def append_info_tag(self, tag: str) -> None:
        """Append an owned tag with its literal text."""
        self._columns.tags.append(tag)
        self._columns.operations.append(INFO_TAG)

    def replace_registers(self, registers: Sequence[KmxRegister]) -> None:
        """Set the register layout separately from the operation sequence."""
        if len(registers) > _MAX_REGISTERS:
            raise ValueError("KMX register count exceeds the format limit")
        updated = []
        for register in registers:
            members = []
            for kind, wire in register.members:
                if not 0 <= wire < _MAX_WIRES:
                    raise ValueError("KMX register operand exceeds the format limit")
                members.append(wire | _REGISTER_QUANTUM if kind == "q" else wire)
            updated.append(_RegisterData(register.name, members))
        self._columns.registers = updated

    def append(self, other: "KmxCircuitBuilder | KmxCircuit") -> None:
        """Append operation records on the same IDs and retain local registers."""
        if isinstance(other, KmxCircuitBuilder):
            self._append_columns(other._columns, reverse=False)
            return
        for record in other:
            if record.tag is not None:
                self.append_info_tag(record.tag)
                continue
            opcode = record.opcode
            if record.condition_bit is not None and opcode != "PUSH_CONDITION":
                opcode += "_IF"
            _append_operation(
                self._columns,
                OPERATION_CODES[opcode],
                record.qubits,
                record.result_bit,
                record.condition_bit,
                record.angle,
            )

    def append_reversed(self, other: "KmxCircuitBuilder") -> None:
        """Append records in reverse order, retaining angle and operand values."""
        self._append_columns(other._columns, reverse=True)

    def _append_columns(self, source: _Columns, *, reverse: bool) -> None:
        destination = self._columns
        destination.operations.extend(
            source.operations[::-1] if reverse else source.operations
        )
        for target, values in zip(destination.words, source.words, strict=True):
            target.extend(reversed(values) if reverse else values)
        destination.angles.extend(reversed(source.angles) if reverse else source.angles)
        destination.tags.extend(reversed(source.tags) if reverse else source.tags)

    def freeze(self) -> KmxCircuit:
        """Validate operand columns and collisions and return an independent value."""
        return _freeze_columns(self._columns)

    def to_kmb(self) -> bytes:
        """Serialize validated columns directly without a KMX formatting pass."""
        circuit = self.freeze()
        return _pack_kmb(self._columns, circuit.qubits, circuit.bits)

    def clear(self) -> None:
        """Remove all operation and register state."""
        self._columns = _Columns()
        self._quantum_storage = WireAllocation()
        self._classical_storage = WireAllocation()
        self._max_qubits = inf


class _BuilderMark:
    def __init__(self, builder: KmxCircuitBuilder, name: str) -> None:
        self._builder = builder
        self._name = name

    def __enter__(self) -> "_BuilderMark":
        self._builder.append_info_tag("@enter:" + self._name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self._builder.append_info_tag("@exit:" + self._name)
        return False
