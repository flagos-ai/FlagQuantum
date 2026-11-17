# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix MutableCircuit.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/mutable_circuit.cc

"""Build raw reversible circuits incrementally and freeze a validated value."""

from collections.abc import Sequence
from itertools import chain
from typing import BinaryIO

from .kmb import (
    _MAX_REGISTERS,
    _MAX_WIRES,
    _REGISTER_QUANTUM,
    KmxRegister,
    _append_kmx_text,
    _append_operation,
    _Columns,
    _RegisterData,
)
from .kmb_operations import INFO_TAG, OPERATION_CODES
from .kmx_circuit import KmxCircuit, _freeze_columns


class KmxCircuitBuilder:
    """Own mutable operation columns with an immutable circuit output boundary."""

    def __init__(self) -> None:
        self._columns = _Columns()

    def __copy__(self) -> "KmxCircuitBuilder":
        duplicate = KmxCircuitBuilder()
        duplicate._columns.registers = [
            _RegisterData(register.name, register.members.copy())
            for register in self._columns.registers
        ]
        duplicate.append(self)
        return duplicate

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

    def clear(self) -> None:
        """Remove all operation and register state."""
        self._columns = _Columns()
