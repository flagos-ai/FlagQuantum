# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked KMX writer's physical DEBUG_PRINT opcodes.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit

"""Preserve physical KMX debug records without adding execution effects."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .._component_trace.trace import ComponentCall
from ..artifacts.reversible_phase import ReversiblePhaseProgram
from .kmx_registers import (
    KmxRegister,
    RegisterMember,
    _is_register_directive,
    decode_kmx_registers,
    encode_kmx_registers,
)

_DEBUG = re.compile(r"DEBUG_PRINT(?:[ \t]+([qb])([0-9]+))?(?:[ \t]+if[ \t]+b([0-9]+))?")


@dataclass(frozen=True)
class KmxDebugRecord:
    """A source debug record preceding a gate or block marker at the given index."""

    position: int
    operand: RegisterMember | None = None
    condition: int | None = None


def _validate_operands(program: ReversiblePhaseProgram, record: KmxDebugRecord) -> None:
    if record.operand is not None:
        kind, index = record.operand
        widths = {"q": program.n_qubits, "b": program.n_bits}
        if (
            kind not in widths
            or type(index) is not int
            or not 0 <= index < widths[kind]
        ):
            raise ValueError("KMX debug operand is out of range")
    if record.condition is not None and (
        type(record.condition) is not int or not 0 <= record.condition < program.n_bits
    ):
        raise ValueError("KMX debug condition is out of range")


def encode_kmx_debug(
    program: ReversiblePhaseProgram,
    registers: Sequence[KmxRegister],
    records: Sequence[KmxDebugRecord],
    *,
    calls: Sequence[ComponentCall] = (),
) -> str:
    """Insert debug records by body index, retaining zero-width marked blocks."""
    lines = encode_kmx_registers(program, registers, calls=calls).split("\n")[:-1]
    start = (
        next(
            index
            for index, line in enumerate(lines)
            if not _is_register_directive(line)
        )
        + 1
    )
    cursor = start
    encoded = lines[:start]
    for record in records:
        if type(
            record.position
        ) is not int or not cursor <= start + record.position <= len(lines):
            raise ValueError("KMX debug positions must be ordered and within the body")
        _validate_operands(program, record)
        stop = start + record.position
        encoded.extend(lines[cursor:stop])
        line = "DEBUG_PRINT"
        if record.operand is not None:
            line += f" {record.operand[0]}{record.operand[1]}"
        if record.condition is not None:
            line += f" if b{record.condition}"
        encoded.append(line)
        cursor = stop
    encoded.extend(lines[cursor:])
    return "\n".join((*encoded, ""))


def decode_kmx_debug(
    text: str,
) -> tuple[
    ReversiblePhaseProgram,
    tuple[KmxRegister, ...],
    tuple[ComponentCall, ...],
    tuple[KmxDebugRecord, ...],
]:
    """Read physical debug records before validating the executable stream."""
    position = -1
    records = []
    body = []
    for line in text.removesuffix("\n").split("\n"):
        match = _DEBUG.fullmatch(line)
        if match is None:
            if line.split(maxsplit=1)[:1] == ["DEBUG_PRINT"]:
                raise ValueError("Invalid physical KMX debug record")
            body.append(line)
            if not _is_register_directive(line):
                position += 1
            continue
        if position < 0:
            raise ValueError("KMX debug record precedes the version header")
        operand = None
        if match[1] is not None:
            kind: Literal["q", "b"] = "q" if match[1] == "q" else "b"
            operand = (kind, int(match[2]))
        condition = int(match[3]) if match[3] is not None else None
        records.append(KmxDebugRecord(position, operand, condition))

    program, registers, calls = decode_kmx_registers("\n".join(body))
    for record in records:
        _validate_operands(program, record)
    return program, registers, calls, tuple(records)
