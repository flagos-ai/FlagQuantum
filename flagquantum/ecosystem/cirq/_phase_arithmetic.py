"""Convert the external phase-addition value contract to existing native IR."""

from collections.abc import Iterator
from typing import Any

from ...compiler.registers.phase.phase_addition import (
    phase_add_ir,
    phase_add_workspace_size,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.signatures import Signature
from ._register_export import export_register_cirq
from .conversion import _cirq


def _phase_add_operations(
    source: Any,
    operands: tuple[Any, ...],
    manager: Any,
    key_prefix: str,
    tag: object,
) -> Iterator[Any]:
    """Realize the source permutation without claiming a missing source recipe.

    Source/phase words are big endian. Work and result slots belong to the
    existing measured-adder owner. Generated keys are private to this capture.
    """
    m, n, shift = source.x_bitsize, source.phase_bitsize, source.right_shift
    value = source.controlled_by
    if any(type(width) is not int or width < 1 for width in (m, n)):
        raise ValueError("Phase addition import requires positive concrete widths")
    if value is not None and (type(value) is not int or value not in (0, 1)):
        raise ValueError("Phase addition import requires a binary control value")
    controlled = value is not None
    count = int(controlled)
    if len(operands) != count + m + n:
        raise ValueError("Phase addition operands do not match source registers")
    required = phase_add_workspace_size(m, n, right_shift=shift, controlled=controlled)
    logical = count + m + n
    scratch = tuple(manager.qalloc(required))
    try:
        gates = phase_add_ir(
            tuple(range(count, count + m)),
            tuple(range(count + m, logical)),
            tuple(range(logical, logical + required)),
            tuple(range(required)),
            right_shift=shift,
            sign=source.sign,
            control=0 if controlled else None,
            control_value=value if value is not None else 1,
        )
        if not gates:
            return
        # The standalone exporter requires every declared wire to occur. Typed
        # through-port placement preserves idle truncated bits, so identity
        # anchors are only transport scaffolding and are removed below.
        used = {wire for gate in gates for wire in gate.wires}
        padding = tuple(
            Instruction("i", (wire,))
            for wire in range(logical + required)
            if wire not in used
        )
        signature = Signature.build(ctrl=count, x=m, phase_grad=n, work=required)
        ports = {
            "ctrl": tuple(range(count)),
            "x": tuple(range(count, count + m)),
            "phase_grad": tuple(range(count + m, logical)),
            "work": tuple(range(logical, logical + required)),
        }
        arrays = {
            "ctrl": operands[:count],
            "x": operands[count : count + m],
            "phase_grad": operands[count + m :],
            "work": scratch,
        }
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
