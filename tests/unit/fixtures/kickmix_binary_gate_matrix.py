"""Read typed operand and effective stream fixtures for locked binary gates."""

import json
from collections.abc import Callable, Iterator
from hashlib import sha256
from pathlib import Path
from typing import TypedDict, cast

from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream

Port = str | list[str]


class Group(TypedDict):
    name: str
    left: list[Port]
    right: list[Port]
    streams: list[list[tuple[int, str]]]


def source_matrix(opcode: str) -> Iterator[tuple[Port, Port, tuple[int, str]]]:
    path = Path(__file__).with_name(f"kickmix_{opcode}.json")
    for group in cast(list[Group], json.loads(path.read_text())):
        for left, row in zip(group["left"], group["streams"], strict=True):
            for right, expected in zip(group["right"], row, strict=True):
                yield left, right, expected


def _identity(token: str) -> OperandIdentity:
    if token in {"0", "1"}:
        return OperandIdentity(OperandKind.BOOLEAN, int(token))
    if token in {"+", "-"}:
        return OperandIdentity(OperandKind.X_BOOLEAN, int(token == "-"))
    if token.startswith("xb"):
        return OperandIdentity(OperandKind.X_CLASSICAL, int(token[2:]))
    kind = {"q": OperandKind.QUANTUM, "b": OperandKind.CLASSICAL}[token[0]]
    return OperandIdentity(kind, int(token[1:]))


def _port(value: Port) -> OperandIdentity | tuple[OperandIdentity, ...]:
    return _identity(value) if isinstance(value, str) else tuple(map(_identity, value))


def verify_stream(
    emit: Callable[[object, object], tuple[Instruction, ...]],
    left: Port,
    right: Port,
    expected: tuple[int, str],
) -> None:
    ports = _port(left), _port(right)
    operands = tuple(
        operand
        for port in ports
        for operand in ((port,) if isinstance(port, OperandIdentity) else port)
    )
    width = 1 + max(
        (item.value for item in operands if item.kind is OperandKind.QUANTUM),
        default=0,
    )
    slots = sorted(
        {
            item.value
            for item in operands
            if item.kind in {OperandKind.CLASSICAL, OperandKind.X_CLASSICAL}
        }
    )
    prefix = tuple(ClassicalXor((), slot).instruction() for slot in slots)
    stream = emit(*ports)
    program = lower_reversible_phase(CircuitIR(width, (*prefix, *stream)))
    found = normalized_stream(
        "\n".join(encode_kmx(program).splitlines()[1 + len(prefix) :])
    )
    assert (len(found), sha256(repr(found).encode()).hexdigest()) == tuple(expected)
