# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix circuit.cc and circuit.h.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/circuit.cc

"""Structural operations on KMX circuits and binary file detection."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain

from .kmb import (
    _CLASSICAL_TARGET,
    _CONDITION,
    _MAGIC,
    _MAX_RECORDS,
    KmxRecord,
    KmxRegister,
    _iter_records,
    _iter_registers,
    _parse_kmx,
    _validated_lines,
    decode_kmb,
)
from .kmb_operations import OPERATION_NAMES, POP, PUSH


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
    columns = _parse_kmx(text)
    return KmxCircuit(tuple(_iter_registers(columns)), tuple(_iter_records(columns)))


def inspect_kmx(text: str) -> KmxCircuitStats:
    """Analyze a structural KMX stream, including its non-executable records."""
    columns = _parse_kmx(text)
    _, qubits, bits = _validated_lines(columns)
    current_depth = max_depth = 0
    for opcode in columns.operations:
        if opcode == PUSH:
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif opcode == POP and current_depth > 0:
            current_depth -= 1
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
        max_condition_depth=max_depth,
        opcode_counts=tuple(sorted(counts.items())),
    )


def concatenate_kmx(left: str, right: str) -> str:
    """Append ordered records on shared wire IDs, preferring the left registers."""
    first = read_kmx_circuit(left)
    second = read_kmx_circuit(right)
    if len(first.records) + len(second.records) > _MAX_RECORDS:
        raise ValueError("KMX circuit exceeds the record limit")
    return KmxCircuit(
        first.registers or second.registers, first.records + second.records
    ).to_kmx()


def repeat_kmx(text: str, repetitions: int) -> str:
    """Repeat ordered records on the same wires and retain the original registers."""
    if not 0 <= repetitions < 1 << 64:
        raise ValueError("KMX repetitions must fit an unsigned 64-bit integer")
    circuit = read_kmx_circuit(text)
    if not circuit.records or repetitions == 1:
        return circuit.to_kmx()
    if len(circuit.records) * repetitions > _MAX_RECORDS:
        raise ValueError("KMX circuit exceeds the record limit")
    return KmxCircuit(circuit.registers, circuit.records * repetitions).to_kmx()


def decode_kmx_or_kmb(data: bytes) -> str:
    """Detect a raw circuit file and return its canonical KMX text."""
    if data.startswith(_MAGIC[:1]):
        if not data.startswith(_MAGIC):
            raise ValueError("Invalid KMB magic")
        return decode_kmb(data)
    lines, _, _ = _validated_lines(_parse_kmx(data.decode(), file_text=True))
    return "\n".join(lines)
