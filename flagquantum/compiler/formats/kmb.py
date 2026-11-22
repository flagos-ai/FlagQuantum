# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix Circuit::write_kmb_to and Circuit::from_kmb_file.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/circuit.cc

"""Encode and decode Kickmix v2 binary packets at the KMX text boundary."""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from struct import Struct, iter_unpack
from typing import Literal

from ..artifacts._operand_tags import MAX_NUM_QUBITS as _MAX_WIRES
from ..artifacts._operand_tags import MAX_NUM_REGISTERS as _MAX_REGISTERS
from ..artifacts._word_arena import WordArena
from ..artifacts.phase_angle import PhaseAngle128
from ._kmx_text import QUOTED_UNIT, escape_kmx_text, unescape_kmx_text
from .kmb_operations import (
    ANGLE,
    BIT_TARGET,
    CONDITION,
    INFO_TAG,
    OPERATION_CODES,
    OPERATION_NAMES,
    PUSH,
    QUANTUM_COLUMNS,
    QUBIT_MASK,
)

_MAGIC = bytes.fromhex("d750c7d5c329d326e3cc9f6834f2b8bf")
_HEADER = Struct("<4I9Q")
_PACKET = Struct("<QQ")
_U32 = Struct("<I")
_U64 = Struct("<Q")
_ANGLE = Struct("<QQ")
_REGISTER = re.compile(rf'REGISTER[ \t]+r([0-9]+)(?:[ \t]+"({QUOTED_UNIT}*)")?')
_MEMBER = re.compile(r"APPEND_TO_REGISTER[ \t]+([qb][0-9]+)[ \t]+r([0-9]+)")
_TAG = re.compile(rf'INFO_TAG[ \t]+"({QUOTED_UNIT}*)"')
_FILE_QUOTED_FIELD = re.compile(r'"(?:[^"\\]|\\.)*"')
_MAX_RECORDS = 2_000_000_000
_CLASSICAL_TARGET, _CONDITION, _PHASE, _TAGS = 8, 9, 10, 11
_REGISTER_QUANTUM = 1 << 31


@dataclass(frozen=True)
class KmxRecord:
    """One validated raw record in physical operand order."""

    opcode: str
    qubits: tuple[int, ...] = ()
    result_bit: int | None = None
    condition_bit: int | None = None
    angle: PhaseAngle128 | None = None
    tag: str | None = None

    def to_kmx(self) -> str:
        """Render the canonical text for this record."""
        if self.tag is not None:
            return f'INFO_TAG "{escape_kmx_text(self.tag)}"'
        name = "DEBUG_PRINT" if self.opcode.startswith("DEBUG_PRINT_") else self.opcode
        fields = [name, *(f"q{qubit}" for qubit in self.qubits)]
        if self.result_bit is not None:
            fields.append(f"b{self.result_bit}")
        if self.angle is not None:
            fields.append(self.angle.to_decimal_half_turns())
        if self.condition_bit is not None:
            fields.extend(("if", f"b{self.condition_bit}"))
        return " ".join(fields)


@dataclass(frozen=True)
class KmxRegister:
    """Raw register name and ordered physical members."""

    name: str
    members: tuple[tuple[Literal["q", "b"], int], ...]

    @property
    def diagnostic_text(self) -> str:
        """Render ordered source register fields with a native type name."""
        contents = "".join(f"{kind}{wire}, " for kind, wire in self.members)
        return f'KmxRegister{{.name="{self.name}", .contents={{{contents}}}}}'

    def to_kmx(self, index: int) -> Iterator[str]:
        """Render ordered members followed by the optional register name."""
        for kind, operand in self.members:
            yield f"APPEND_TO_REGISTER {kind}{operand} r{index}"
        if self.name:
            yield f'REGISTER r{index} "{escape_kmx_text(self.name)}"'
        elif not self.members:
            yield f"REGISTER r{index}"


@dataclass
class _RegisterData:
    name: str = ""
    members: list[int] = field(default_factory=list)


@dataclass
class _Columns:
    registers: list[_RegisterData] = field(default_factory=list)
    operations: bytearray = field(default_factory=bytearray)
    words: tuple[WordArena, ...] = field(
        default_factory=lambda: tuple(WordArena() for _ in range(10))
    )
    angles: list[PhaseAngle128] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def register(self, index: int) -> _RegisterData:
        if index >= _MAX_REGISTERS:
            raise ValueError("KMB register index exceeds the format limit")
        self.registers.extend(
            _RegisterData() for _ in range(index + 1 - len(self.registers))
        )
        return self.registers[index]


class _Reader:
    def __init__(self, data: bytes | memoryview) -> None:
        self.data = memoryview(data)
        self.position = 0

    def take(self, size: int) -> memoryview:
        end = self.position + size
        if end > len(self.data):
            raise ValueError("Truncated KMB payload")
        start = self.position
        self.position = end
        return self.data[start:end]

    def packet(self, expected: int, length: int | None = None) -> memoryview:
        kind, actual = _PACKET.unpack(self.take(_PACKET.size))
        if kind != expected or (length is not None and length != actual):
            raise ValueError(
                f"Invalid KMB packet order or length for packet {expected}"
            )
        return self.take(actual)

    def finish(self) -> None:
        if self.position != len(self.data):
            raise ValueError("Unexpected trailing KMB data")


def _index(token: str, prefix: str) -> int:
    if not re.fullmatch(prefix + r"[0-9]+", token):
        raise ValueError(f"Invalid KMX operand {token!r}")
    index = int(token[1:])
    if index >= _MAX_WIRES:
        raise ValueError("KMB operand index exceeds the format limit")
    return index


def _parse_statement(columns: _Columns, line: str) -> None:
    if tag := _TAG.fullmatch(line):
        columns.operations.append(INFO_TAG)
        columns.tags.append(unescape_kmx_text(tag[1]))
        return
    if register := _REGISTER.fullmatch(line):
        columns.register(int(register[1])).name = unescape_kmx_text(register[2] or "")
        return
    if member := _MEMBER.fullmatch(line):
        token, index = member.groups()
        value = _index(token, "[qb]")
        if token.startswith("q"):
            value |= _REGISTER_QUANTUM
        columns.register(int(index)).members.append(value)
        return
    opcode, *ports = line.split()
    condition = None
    if len(ports) >= 2 and ports[-2] == "if":
        condition = _index(ports[-1], "b")
        del ports[-2:]
    if opcode == "DEBUG_PRINT":
        _append_debug(columns, ports, condition)
        return
    if opcode.startswith("DEBUG_PRINT_") or opcode.endswith("_IF"):
        raise ValueError(f"Invalid KMX operation spelling {opcode}")
    suffix = "_IF" if condition is not None and opcode != "PUSH_CONDITION" else ""
    code = OPERATION_CODES.get(opcode + suffix)
    if (
        code is None
        or code == INFO_TAG
        or bool(code & CONDITION) != (condition is not None)
    ):
        raise ValueError(f"Invalid KMX operation {line!r}")
    quantum = QUANTUM_COLUMNS[(code & QUBIT_MASK) >> 4]
    has_result = bool(code & BIT_TARGET)
    has_angle = (code & ~CONDITION) == ANGLE
    if len(ports) != len(quantum) + has_result + has_angle:
        raise ValueError(f"Invalid KMX operand count for {opcode}")
    # Complete syntax checks before appending any column of the statement.
    operands = tuple(_index(token, "q") for token in ports[: len(quantum)])
    result = _index(ports[len(quantum)], "b") if has_result else None
    angle = (
        PhaseAngle128.from_rounded_decimal_half_turns(ports[-1]) if has_angle else None
    )
    _append_operation(columns, code, operands, result, condition, angle)


def _append_debug(
    columns: _Columns, ports: Sequence[str], condition: int | None
) -> None:
    suffix = "_IF" if condition is not None else ""

    def append_member(kind: str, index: int) -> None:
        if kind == "q":
            _append_operation(
                columns,
                OPERATION_CODES["DEBUG_PRINT_Q" + suffix],
                (index,),
                None,
                condition,
                None,
            )
        else:
            _append_operation(
                columns,
                OPERATION_CODES["DEBUG_PRINT_C" + suffix],
                (),
                index,
                condition,
                None,
            )

    if not ports:
        _append_operation(
            columns,
            OPERATION_CODES["DEBUG_PRINT_EMPTY" + suffix],
            (),
            None,
            condition,
            None,
        )
    for token in ports:
        kind = token[:1]
        if kind in {"q", "b"}:
            append_member(kind, _index(token, kind))
        elif kind == "r":
            index = _index(token, "r")
            if index >= _MAX_REGISTERS:
                raise ValueError("KMB register index exceeds the format limit")
            if index < len(columns.registers):
                for member in columns.registers[index].members:
                    append_member(
                        "q" if member & _REGISTER_QUANTUM else "b",
                        member & ~_REGISTER_QUANTUM,
                    )
        else:
            raise ValueError(f"Invalid KMX debug operand {token!r}")


def _append_operation(
    columns: _Columns,
    code: int,
    qubits: tuple[int, ...],
    result: int | None,
    condition: int | None,
    angle: PhaseAngle128 | None,
) -> None:
    for packet, qubit in zip(
        QUANTUM_COLUMNS[(code & QUBIT_MASK) >> 4], qubits, strict=True
    ):
        columns.words[packet].append(qubit)
    if result is not None:
        columns.words[_CLASSICAL_TARGET].append(result)
    if condition is not None:
        columns.words[_CONDITION].append(condition)
    if angle is not None:
        columns.angles.append(angle)
    columns.operations.append(code)


def _iter_registers(columns: _Columns) -> Iterator[KmxRegister]:
    names = [reg.name for reg in columns.registers if reg.name]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate KMB register name")
    for register in columns.registers:
        yield KmxRegister(
            register.name,
            tuple(
                (
                    "q" if member & _REGISTER_QUANTUM else "b",
                    member & ~_REGISTER_QUANTUM,
                )
                for member in register.members
            ),
        )


def _register_lines(columns: _Columns) -> tuple[list[str], int, int]:
    lines: list[str] = []
    q_max = -1
    b_max = -1
    for index, register in enumerate(_iter_registers(columns)):
        for kind, operand in register.members:
            if kind == "q":
                q_max = max(q_max, operand)
            else:
                b_max = max(b_max, operand)
        lines.extend(register.to_kmx(index))

    return lines, q_max, b_max


def _iter_records(columns: _Columns) -> Iterator[KmxRecord]:
    cursors = [0] * len(columns.words)

    def read(packet: int) -> int:
        cursor = cursors[packet]
        if cursor >= len(columns.words[packet]):
            raise ValueError(f"KMB operand packet {packet} underflow")
        cursors[packet] += 1
        return columns.words[packet][cursor]

    tag_index = angle_index = 0
    for code in columns.operations:
        if code not in OPERATION_NAMES:
            raise ValueError(f"Unknown KMB operation byte {code:#04x}")
        if code == INFO_TAG:
            if tag_index >= len(columns.tags):
                raise ValueError("KMB tag packet underflow")
            yield KmxRecord("INFO_TAG", tag=columns.tags[tag_index])
            tag_index += 1
            continue
        base = code if code == PUSH else code & ~CONDITION
        operands = tuple(
            read(packet) for packet in QUANTUM_COLUMNS[(code & QUBIT_MASK) >> 4]
        )
        if len(set(operands)) != len(operands):
            raise ValueError("KMB operation repeats a quantum wire")
        target = read(_CLASSICAL_TARGET) if code & BIT_TARGET else None
        angle = None
        if base == ANGLE:
            if angle_index >= len(columns.angles):
                raise ValueError("KMB angle packet underflow")
            angle = columns.angles[angle_index]
            angle_index += 1
        condition = read(_CONDITION) if code & CONDITION else None
        yield KmxRecord(OPERATION_NAMES[base], operands, target, condition, angle)
    if cursors != list(map(len, columns.words)) or (tag_index, angle_index) != (
        len(columns.tags),
        len(columns.angles),
    ):
        raise ValueError("KMB operation column counts do not match the opcode stream")


def _validated_lines(columns: _Columns) -> tuple[list[str], int, int]:
    lines, q_max, b_max = _register_lines(columns)
    for record in _iter_records(columns):
        q_max = max(q_max, max(record.qubits, default=-1))
        if record.result_bit is not None:
            b_max = max(b_max, record.result_bit)
        if record.condition_bit is not None:
            b_max = max(b_max, record.condition_bit)
        lines.append(record.to_kmx())
    if max(q_max, b_max) >= _MAX_WIRES:
        raise ValueError("KMB circuit exceeds the operand limit")
    return lines, q_max + 1, b_max + 1


def _packet(kind: int, payload: bytes) -> bytes:
    return _PACKET.pack(kind, len(payload)) + payload


def _pack_words(values: Sequence[int], word: Struct) -> bytes:
    return b"".join(word.pack(value) for value in values)


def _parse_kmx(text: str, *, file_text: bool = False) -> _Columns:
    columns = _Columns()
    _append_kmx_text(columns, text, file_text=file_text)
    return columns


def _append_kmx_text(columns: _Columns, text: str, *, file_text: bool) -> None:
    for raw in re.split(r"[\r\n]", text):
        if file_text:
            raw = _FILE_QUOTED_FIELD.sub(
                lambda match: match[0].replace("#", r"\P"), raw, count=1
            )
        if line := raw.partition("#")[0].strip():
            _parse_statement(columns, line)


def encode_kmb(text: str, *, file_text: bool = False) -> bytes:
    """Encode KMX operations and register directives as source-compatible KMB v2."""
    columns = _parse_kmx(text, file_text=file_text)
    _, n_qubits, n_bits = _validated_lines(columns)
    return _pack_kmb(columns, n_qubits, n_bits)


def _pack_kmb(columns: _Columns, n_qubits: int, n_bits: int) -> bytes:
    names = [tag.encode() for tag in columns.tags]
    tag_text = b"".join(names)
    offsets = []
    offset = 0
    for name in names:
        offset += len(name)
        offsets.append(offset)
    if max(len(columns.operations), len(tag_text)) > _MAX_RECORDS:
        raise ValueError("KMB circuit exceeds the record limit")
    counts = [len(columns.words[k]) for k in (2, 5, 7, 8, 9)]
    values = [
        _MAGIC,
        _HEADER.pack(
            2,
            n_qubits,
            n_bits,
            len(columns.registers),
            len(columns.operations),
            *counts,
            len(columns.angles),
            len(columns.tags),
            len(tag_text),
        ),
    ]
    for register in columns.registers:
        name = register.name.encode()
        payload = _U32.pack(len(name)) + _U32.pack(len(register.members))
        values.append(_packet(0, payload + name + _pack_words(register.members, _U32)))
    values.append(_packet(1, bytes(columns.operations)))
    values.extend(
        _packet(kind, _pack_words(columns.words[kind], _U32)) for kind in range(2, 10)
    )
    values.append(
        _packet(_PHASE, b"".join(_ANGLE.pack(a.low, a.high) for a in columns.angles))
    )
    values.append(
        _packet(_TAGS, _U64.pack(len(offsets)) + _pack_words(offsets, _U64) + tag_text)
    )
    return b"".join(values)


def decode_kmb(data: bytes) -> str:
    """Decode KMB v2 to canonical KMX after validating all packet lengths."""
    columns, widths = _read_kmb_columns(data)
    lines, actual_q, actual_b = _validated_lines(columns)
    if widths != (actual_q, actual_b):
        raise ValueError("KMB width does not match the referenced operands")
    return "\n".join(lines)


def _read_kmb_columns(data: bytes) -> tuple[_Columns, tuple[int, int]]:
    reader = _Reader(data)
    if reader.take(len(_MAGIC)) != _MAGIC:
        raise ValueError("Invalid KMB magic")
    version, q, b, register_count, ops, q3, q2, q1, bt, bc, angles, tags, chars = (
        _HEADER.unpack(reader.take(_HEADER.size))
    )
    if version != 2:
        raise ValueError("Unsupported KMB version")
    if ops > _MAX_RECORDS:
        raise ValueError(f"KMB num_ops={ops} > MAX_CIRCUIT_INSTRUCTIONS={_MAX_RECORDS}")
    if (
        max(q, b) > _MAX_WIRES
        or register_count > _MAX_REGISTERS
        or chars > _MAX_RECORDS
    ):
        raise ValueError("KMB header exceeds format limits")
    if max(q3, q2, q1, bt, bc, angles, tags) > ops:
        raise ValueError("KMB operation column count exceeds opcode count")
    columns = _Columns()
    for _ in range(register_count):
        register = _Reader(reader.packet(0))
        length = _U32.unpack(register.take(4))[0]
        count = _U32.unpack(register.take(4))[0]
        name = bytes(register.take(length)).decode()
        members = [value for (value,) in iter_unpack("<I", register.take(count * 4))]
        register.finish()
        columns.registers.append(_RegisterData(name, members))
    columns.operations.extend(reader.packet(1, ops))
    for kind, count in enumerate((q3, q3, q3, q2, q2, q1, bt, bc), start=2):
        columns.words[kind].extend(
            value for (value,) in iter_unpack("<I", reader.packet(kind, count * 4))
        )
    columns.angles.extend(
        PhaseAngle128(low, high)
        for low, high in iter_unpack("<QQ", reader.packet(_PHASE, angles * 16))
    )
    tag_packet = _Reader(reader.packet(_TAGS, (tags + 1) * 8 + chars))
    if _U64.unpack(tag_packet.take(8))[0] != tags:
        raise ValueError("KMB tag count does not match the header")
    offsets = [value for (value,) in iter_unpack("<Q", tag_packet.take(tags * 8))]
    tag_data = tag_packet.take(chars)
    last = 0
    for stop in offsets:
        if not last <= stop <= chars:
            raise ValueError("Invalid KMB tag offset")
        columns.tags.append(bytes(tag_data[last:stop]).decode())
        last = stop
    if last != chars:
        raise ValueError("Final KMB tag offset does not match the header")
    reader.finish()
    return columns, (q, b)
