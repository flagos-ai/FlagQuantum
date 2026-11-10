"""Compose bounded matrix QPE through existing controlled-power effects."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .frontend_matrix import capture_unitary
from .frontend_python import _error
from .frontend_storage import require_root_storage
from .program_types import unitary
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_matrix_qpe(owner: _Body, node: ast.Call) -> None:
    """Keep numerical phase formats separate from raw mutable helper borrowing."""
    require_root_storage(owner, node)
    if len(node.args) != 2:
        _error(
            owner.source, node, "program.qpe", "qpe takes a callback and phase counter"
        )
    callback, counter_node = node.args
    if not isinstance(callback, ast.Lambda):
        _error(
            owner.source,
            callback,
            "program.qpe",
            "use a static zero-argument lambda containing one unitary call",
        )
    args, body = callback.args, callback.body
    if (
        args.posonlyargs
        or args.args
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or not isinstance(body, ast.Call)
        or body.keywords
        or any(isinstance(a, ast.Starred) for a in body.args)
    ):
        _error(
            owner.source,
            callback,
            "program.qpe",
            "use one positional unitary call in a zero-argument lambda",
        )
    if owner.resolve_callable(body.func) is not unitary:
        _error(
            owner.source,
            body,
            "program.qpe",
            "the admitted qpe callback contains one static unitary call",
        )
    if not isinstance(counter_node, ast.Name) or counter_node.id not in owner.ports:
        _error(
            owner.source,
            counter_node,
            "program.qpe",
            "qpe requires one whole all-fractional numeric counter",
        )
    port = owner.ports[counter_node.id]
    counter = owner.register(counter_node)
    width = register_width(counter)
    if (
        port.kind != "qnum"
        or port.role == "const"
        or not 1 <= width <= 8
        or port.fractional_bits != width
    ):
        _error(
            owner.source,
            counter_node,
            "program.qpe",
            "qpe requires one to eight writable all-fractional numeric bits",
        )
    previous = next(
        (op for op in reversed(owner.operations) if counter in op.operands[1:]), None
    )
    if previous is None or previous.name != "qreg.allocate":
        _error(
            owner.source,
            counter_node,
            "program.qpe",
            "phase counter must be initialized by allocate and unused",
        )
    start, effect = len(owner.operations), owner.effect
    capture_unitary(owner, body)
    operation = owner.operations.pop()
    owner.effect = effect
    assert len(owner.operations) == start and operation.name == "qreg.unitary"
    data = operation.operands[1:]
    if counter in data:
        _error(
            owner.source,
            body,
            "program.qpe",
            "phase counter and callback data must be disjoint",
        )
    for bit in range(width):
        owner.emit("qreg.gate", node, (counter,), gate="h", index=bit)
    for bit in range(width):
        owner.emit(
            "qreg.unitary",
            node,
            (*data, counter),
            **dict(operation.attributes),
            control_index=bit,
            power=1 << bit,
        )
    owner.emit("qreg.inverse_qft", node, (counter,))
