"""Realize the external square value contract with the owned column compressor."""

from collections.abc import Iterator
from typing import Any

from ...compiler.registers.arithmetic.integer_square import (
    integer_square_ir,
    integer_square_workspace_size,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.signatures import Signature
from ._register_export import export_register_cirq
from .conversion import _cirq


def _square_operations(
    source: Any,
    operands: tuple[Any, ...],
    manager: Any,
    key_prefix: str,
    tag: object,
) -> Iterator[Any]:
    """Use the source's unsigned-square contract, not an absent source recipe.

    The forward result is RIGHT and begins at zero. Its adjoint consumes the
    matching result. The native XOR square implements both on these domains;
    its paper-based construction and measured work remain separately owned.
    """
    width = source.bitsize
    if type(width) is not int or width < 1:
        raise ValueError("Square import requires a positive concrete width")
    logical = 3 * width
    if len(operands) != logical:
        raise ValueError("Square operands do not match its input and result")
    required = integer_square_workspace_size(width)
    scratch = tuple(manager.qalloc(required))
    try:
        gates = integer_square_ir(
            tuple(range(width)),
            tuple(range(width, logical)),
            tuple(range(logical, logical + required)),
            measurement_bit=0,
        )
        used = {wire for gate in gates for wire in gate.wires}
        padding = tuple(
            Instruction("i", (wire,))
            for wire in range(logical + required)
            if wire not in used
        )
        signature = Signature.build(a=width, result=2 * width, work=required)
        ports = {
            "a": tuple(range(width)),
            "result": tuple(range(width, logical)),
            "work": tuple(range(logical, logical + required)),
        }
        arrays = {"a": operands[:width], "result": operands[width:], "work": scratch}
        names = {reg.name for reg in signature}
        exported, _ = export_register_cirq(
            CircuitIR(logical + required, (*gates, *padding)),
            signature,
            {name: value for name, value in ports.items() if name in names},
            input_quregs={
                name: value for name, value in arrays.items() if name in names
            },
            feedback_key_prefix=key_prefix,
        )
        cirq = _cirq()
        yield from (
            op.with_tags(tag)
            for op in exported.circuit.all_operations()
            if op.gate != cirq.I
        )
    finally:
        manager.qfree(scratch)
