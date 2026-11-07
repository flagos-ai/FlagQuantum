"""KMX codec for the versioned reversible-phase operation subset."""

import re

from ..artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)

_HEADER = (
    'INFO_TAG "@flagquantum:reversible-phase-v1 wires={wires} bits={bits}{inputs}"'
)
_HEADER_PATTERN = re.compile(
    r'INFO_TAG "@flagquantum:reversible-phase-v1 wires=([0-9]+) bits=([0-9]+)(?: inputs=([0-9]+(?:,[0-9]+)*))?"'
)
_OPERAND = re.compile(r"[qb][0-9]+")


def encode_kmx(program: ReversiblePhaseProgram) -> str:
    """Encode operations and preserve declared register widths in a native info tag."""
    inputs = (
        f" inputs={','.join(map(str, program.input_bits))}"
        if program.input_bits
        else ""
    )
    lines = [_HEADER.format(wires=program.n_qubits, bits=program.n_bits, inputs=inputs)]
    for operation in program.operations:
        fields = [operation.opcode.upper(), *(f"q{wire}" for wire in operation.qubits)]
        if operation.result_bit is not None:
            fields.append(f"b{operation.result_bit}")
        if operation.condition_bit is not None:
            fields.extend(("if", f"b{operation.condition_bit}"))
        lines.append(" ".join(fields))
    return "".join(f"{line}\n" for line in lines)


def decode_kmx(text: str) -> ReversiblePhaseProgram:
    """Decode a tagged stream from the supported subset, including unused slots."""
    lines = text.splitlines()
    header = _HEADER_PATTERN.fullmatch(lines[0]) if lines else None
    if header is None:
        raise ValueError("Missing reversible-phase KMX version and register widths")
    wires, bits = map(int, header.groups()[:2])
    inputs = tuple(map(int, header[3].split(","))) if header[3] else ()
    operations = []
    for line in lines[1:]:
        opcode, *operands = line.split()
        condition = None
        if len(operands) >= 2 and operands[-2] == "if":
            condition = _parse_operand(operands[-1], "b")
            del operands[-2:]
        result = None
        if opcode in {"HMR", "BIT_STORE0", "BIT_INVERT"} and operands:
            result = _parse_operand(operands.pop(), "b")
        operations.append(
            ReversiblePhaseOperation(
                opcode.lower(),
                tuple(_parse_operand(operand, "q") for operand in operands),
                result_bit=result,
                condition_bit=condition,
            )
        )
    return ReversiblePhaseProgram(wires, bits, tuple(operations), inputs)


def _parse_operand(text: str, prefix: str) -> int:
    if not text.startswith(prefix) or _OPERAND.fullmatch(text) is None:
        raise ValueError(f"Invalid KMX operand {text!r}")
    return int(text[1:])
