"""Expand verified callback arithmetic using closed scratch and shared controls."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .._hybrid.model import Operation, Value, ValueId
from .._hybrid.unitary_effects import unitary_regions
from .qpe_control import emit_controlled_effect
from .register_profile import register_type, register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def emit_callback_effects(
    owner: _Body,
    node: ast.Call,
    effects: Sequence[Operation],
    predicate: Value,
    index: int,
) -> None:
    """Keep scratch computation unconditional and control only its pure action.

    The caller verifies Const predicate inputs and the captured unitary regions
    before expansion. A word XOR computes a disjoint temporary, controls its XOR
    into the arbitrary target, then uncomputes and releases the temporary.
    """

    def emit(
        name: str, values: tuple[Value, ...], attributes: dict[str, object]
    ) -> None:
        owner.emit(name, node, values, **attributes)

    def controlled(operation: Operation) -> None:
        if operation.name != "qreg.xor":
            emit_controlled_effect(predicate, index, operation, emit)
            return
        target, *arguments = operation.operands[1:]
        temporary = Value(
            ValueId(owner.name, owner.counter), register_type(register_width(target))
        )
        owner.counter += 1
        owner.operations.append(
            Operation(
                "qreg.local",
                results=(temporary,),
                attributes={"name": f"callback.{temporary.id.index}"},
                location=owner.source.location(node),
            )
        )
        owner.emit(
            "qreg.compute",
            node,
            (temporary, *arguments),
            callee=operation.attributes["callee"],
        )
        for bit in range(register_width(target)):
            owner.emit(
                "qreg.control_x",
                node,
                (predicate, temporary, target),
                indices=(index, bit, bit),
            )
        owner.emit("qreg.uncompute", node, (temporary,))
        owner.emit("qreg.release", node, (temporary,))

    for region in unitary_regions(effects):
        if len(region) == 1:
            controlled(region[0])
            continue
        owner.operations.append(region[0])
        compute = region[1]
        owner.emit(compute.name, node, compute.operands[1:], **compute.attributes)
        emit_callback_effects(owner, node, region[2:-2], predicate, index)
        for operation in region[-2:]:
            owner.emit(
                operation.name, node, operation.operands[1:], **operation.attributes
            )
