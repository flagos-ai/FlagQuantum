# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix circuit.cc and circuit.h.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/circuit.cc

"""Structural operations on KMX circuits and binary file detection."""

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from operator import index as integer_index
from types import NotImplementedType
from typing import SupportsIndex, overload

from ..artifacts.operand_array import OperandArray
from ..artifacts.operand_identity import OperandIdentity, OperandKind
from .kmb import (
    _CLASSICAL_TARGET,
    _CONDITION,
    _MAGIC,
    _MAX_RECORDS,
    KmxRecord,
    KmxRegister,
    _Columns,
    _iter_records,
    _iter_registers,
    _parse_kmx,
    _read_kmb_columns,
    _validated_lines,
)
from .kmb_operations import OPERATION_NAMES


@dataclass(frozen=True)
class KmxCircuitStats:
    """Counts of serialized records and inferred wire widths."""

    qubits: int
    bits: int
    registers: int
    records: int
    quantum_columns: tuple[int, int, int]
    classical_targets: int
    conditions: int
    rotations: int
    tags: int
    tag_bytes: int
    max_condition_depth: int
    opcode_counts: tuple[tuple[str, int], ...]

    @property
    def ccz_state_weight(self) -> int:
        """Return the source estimate of CCZ states needed for these records."""
        return self.quantum_columns[2] + (self.rotations + 1) // 2


@dataclass(frozen=True)
class KmxCircuit:
    """Validated records and raw register declarations for format consumers."""

    registers: tuple[KmxRegister, ...]
    records: tuple[KmxRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def register_data(self) -> dict[str | OperandIdentity, OperandArray]:
        """Return independent named or indexed projections of register contents."""
        return {
            register.name
            or OperandIdentity(OperandKind.REGISTER, index): OperandArray(
                OperandIdentity(
                    OperandKind.QUANTUM if kind == "q" else OperandKind.CLASSICAL,
                    value,
                )
                for kind, value in register.members
            )
            for index, register in enumerate(self.registers)
        }

    @overload
    def __add__(self, other: "KmxCircuit") -> "KmxCircuit": ...

    @overload
    def __add__(self, other: object) -> "KmxCircuit | NotImplementedType": ...

    def __add__(self, other: object) -> "KmxCircuit | NotImplementedType":
        if not isinstance(other, KmxCircuit):
            return NotImplemented
        if len(self.records) + len(other.records) > _MAX_RECORDS:
            raise ValueError("KMX circuit exceeds the record limit")
        return KmxCircuit(
            self.registers or other.registers, self.records + other.records
        )

    @overload
    def __mul__(self, repetitions: SupportsIndex) -> "KmxCircuit": ...

    @overload
    def __mul__(self, repetitions: object) -> "KmxCircuit | NotImplementedType": ...

    def __mul__(self, repetitions: object) -> "KmxCircuit | NotImplementedType":
        if not isinstance(repetitions, SupportsIndex):
            return NotImplemented
        count = integer_index(repetitions)
        if not -(1 << 63) <= count < 1 << 63:
            raise TypeError("Circuit repetitions must fit a signed 64-bit integer")
        return _repeat_circuit(self, max(0, count))

    @overload
    def __rmul__(self, repetitions: SupportsIndex) -> "KmxCircuit": ...

    @overload
    def __rmul__(self, repetitions: object) -> "KmxCircuit | NotImplementedType": ...

    def __rmul__(self, repetitions: object) -> "KmxCircuit | NotImplementedType":
        return self.__mul__(repetitions)

    @property
    def qubits(self) -> int:
        """Return the highest physical qubit identifier plus one."""
        members = (
            index
            for register in self.registers
            for kind, index in register.members
            if kind == "q"
        )
        operands = (qubit for record in self.records for qubit in record.qubits)
        return max(chain(members, operands), default=-1) + 1

    @property
    def bits(self) -> int:
        """Return the highest physical bit identifier plus one."""
        members = (
            index
            for register in self.registers
            for kind, index in register.members
            if kind == "b"
        )
        operands = (
            bit
            for record in self.records
            for bit in (record.result_bit, record.condition_bit)
            if bit is not None
        )
        return max(chain(members, operands), default=-1) + 1

    def __iter__(self) -> Iterator[KmxRecord]:
        return iter(self.records)

    def max_condition_depth(self) -> int:
        """Measure condition nesting on the frozen physical stream."""
        return _condition_depth(record.opcode for record in self)

    def to_kmx(self) -> str:
        """Render register declarations and ordered records without a final newline."""
        register_lines = (
            line
            for index, register in enumerate(self.registers)
            for line in register.to_kmx(index)
        )
        return "\n".join(chain(register_lines, (record.to_kmx() for record in self)))


def read_kmx_circuit(text: str) -> KmxCircuit:
    """Read raw records, register membership and the inferred quantum width."""
    return _freeze_columns(_parse_kmx(text))


def read_kmb_circuit(data: bytes) -> KmxCircuit:
    """Read KMB directly to immutable physical records without rendering text."""
    columns, widths = _read_kmb_columns(data)
    circuit = _freeze_columns(columns)
    if widths != (circuit.qubits, circuit.bits):
        raise ValueError("KMB width does not match the referenced operands")
    return circuit


def _freeze_columns(columns: _Columns) -> KmxCircuit:
    if len(columns.operations) > _MAX_RECORDS:
        raise ValueError("KMX circuit exceeds the record limit")
    return KmxCircuit(tuple(_iter_registers(columns)), tuple(_iter_records(columns)))


def _condition_depth(opcodes: Iterable[str]) -> int:
    current_depth = max_depth = 0
    for opcode in opcodes:
        if opcode == "PUSH_CONDITION":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif opcode == "POP_CONDITION" and current_depth > 0:
            current_depth -= 1
    return max_depth


def inspect_kmx(text: str) -> KmxCircuitStats:
    """Analyze a structural KMX stream, including its non-executable records."""
    columns = _parse_kmx(text)
    _, qubits, bits = _validated_lines(columns)
    counts = Counter(OPERATION_NAMES[opcode] for opcode in columns.operations)
    return KmxCircuitStats(
        qubits=qubits,
        bits=bits,
        registers=len(columns.registers),
        records=len(columns.operations),
        quantum_columns=(
            len(columns.words[7]),
            len(columns.words[5]),
            len(columns.words[2]),
        ),
        classical_targets=len(columns.words[_CLASSICAL_TARGET]),
        conditions=len(columns.words[_CONDITION]),
        rotations=len(columns.angles),
        tags=len(columns.tags),
        tag_bytes=sum(len(tag.encode()) for tag in columns.tags),
        max_condition_depth=_condition_depth(
            OPERATION_NAMES[opcode] for opcode in columns.operations
        ),
        opcode_counts=tuple(sorted(counts.items())),
    )


def concatenate_kmx(left: str, right: str) -> str:
    """Append ordered records on shared wire IDs, preferring the left registers."""
    first = read_kmx_circuit(left)
    second = read_kmx_circuit(right)
    return (first + second).to_kmx()


def repeat_kmx(text: str, repetitions: int) -> str:
    """Repeat ordered records on the same wires and retain the original registers."""
    if not 0 <= repetitions < 1 << 64:
        raise ValueError("KMX repetitions must fit an unsigned 64-bit integer")
    circuit = read_kmx_circuit(text)
    return _repeat_circuit(circuit, repetitions).to_kmx()


def _repeat_circuit(circuit: KmxCircuit, repetitions: int) -> KmxCircuit:
    if not circuit.records or repetitions == 1:
        return circuit
    if len(circuit.records) * repetitions > _MAX_RECORDS:
        raise ValueError("KMX circuit exceeds the record limit")
    return KmxCircuit(circuit.registers, circuit.records * repetitions)


def decode_kmx_or_kmb(data: bytes) -> str:
    """Detect a raw circuit file and return its canonical KMX text."""
    return read_circuit_file(data).to_kmx()


def read_circuit_file(data: bytes) -> KmxCircuit:
    """Detect the raw file format and read immutable physical records."""
    if data.startswith(_MAGIC[:1]):
        if not data.startswith(_MAGIC):
            raise ValueError("Invalid KMB magic")
        return read_kmb_circuit(data)
    return _freeze_columns(_parse_kmx(data.decode(), file_text=True))
