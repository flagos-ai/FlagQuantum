# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked KMX Circuit writer and MutableCircuit register parser.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit

"""KMX register declarations alongside tagged executable operations."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .._component_trace.trace import ComponentCall
from ..artifacts.reversible_phase import ReversiblePhaseProgram
from ._kmx_text import QUOTED_UNIT, escape_kmx_text, unescape_kmx_text
from .kmx_block_marks import decode_kmx_block_marks, encode_kmx_block_marks

_REGISTER = re.compile(rf'REGISTER[ \t]+r([0-9]+)(?:[ \t]+"({QUOTED_UNIT}*)")?')
_APPEND = re.compile(r"APPEND_TO_REGISTER[ \t]+([qb])([0-9]+)[ \t]+r([0-9]+)")
_MAX_REGISTERS = 1_000_000
_MAX_MEMBERS_PER_KIND = 500_000_000

RegisterMember: TypeAlias = tuple[Literal["q", "b"], int]


@dataclass(frozen=True)
class KmxRegister:
    """Physical members in declaration order, with an optional display name."""

    name: str = ""
    members: tuple[RegisterMember, ...] = ()


def _validate_registers(
    program: ReversiblePhaseProgram, registers: Sequence[KmxRegister]
) -> None:
    if len(registers) > _MAX_REGISTERS:
        raise ValueError("KMX register ID is out of range")
    names: set[str] = set()
    widths = {"q": program.n_qubits, "b": program.n_bits}
    for register in registers:
        if register.name:
            if register.name in names:
                raise ValueError(f"Duplicate register name: {register.name}")
            names.add(register.name)
        for kind, index in register.members:
            if (
                kind not in widths
                or type(index) is not int
                or not 0 <= index < min(widths[kind], _MAX_MEMBERS_PER_KIND)
            ):
                raise ValueError(f"KMX register member is out of range: {kind}{index}")


def encode_kmx_registers(
    program: ReversiblePhaseProgram,
    registers: Sequence[KmxRegister],
    *,
    calls: Sequence[ComponentCall] = (),
) -> str:
    """Emit dense IDs by position, ordering source membership before names."""
    _validate_registers(program, registers)
    lines: list[str] = []
    for index, register in enumerate(registers):
        lines.extend(
            f"APPEND_TO_REGISTER {kind}{member} r{index}\n"
            for kind, member in register.members
        )
        if register.name:
            name = escape_kmx_text(register.name)
            lines.append(f'REGISTER r{index} "{name}"\n')
        elif not register.members:
            lines.append(f"REGISTER r{index}\n")
    lines.append(encode_kmx_block_marks(program, calls))
    return "".join(lines)


def decode_kmx_registers(
    text: str,
) -> tuple[ReversiblePhaseProgram, tuple[KmxRegister, ...], tuple[ComponentCall, ...]]:
    """Gather source declarations without synthesizing executable operations."""
    names: dict[int, str] = {}
    members: dict[int, list[RegisterMember]] = {}
    body = []
    for line in text.removesuffix("\n").split("\n"):
        declaration = _REGISTER.fullmatch(line)
        append = _APPEND.fullmatch(line)
        if declaration is not None:
            index = int(declaration[1])
            if index >= _MAX_REGISTERS:
                raise ValueError("KMX register ID is out of range")
            names[index] = unescape_kmx_text(declaration[2] or "")
        elif append is not None:
            index = int(append[3])
            if index >= _MAX_REGISTERS:
                raise ValueError("KMX register ID is out of range")
            kind: Literal["q", "b"] = "q" if append[1] == "q" else "b"
            members.setdefault(index, []).append((kind, int(append[2])))
        elif line.split(maxsplit=1)[:1] in (["REGISTER"], ["APPEND_TO_REGISTER"]):
            raise ValueError("Invalid KMX register declaration")
        else:
            body.append(line)
    count = max(max(names, default=-1), max(members, default=-1)) + 1
    empty = KmxRegister()
    registers = tuple(
        (
            KmxRegister(names.get(index, ""), tuple(members.get(index, ())))
            if index in names or index in members
            else empty
        )
        for index in range(count)
    )
    program, calls = decode_kmx_block_marks("\n".join(body))
    _validate_registers(program, registers)
    return program, registers, calls
