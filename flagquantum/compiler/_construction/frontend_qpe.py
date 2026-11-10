"""Compose bounded pure QPE callbacks through shared controlled effects."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.model import Value
from .frontend_evolution import flush_evolutions
from .frontend_phase import capture_polynomial_phase
from .frontend_python import _error
from .frontend_storage import require_root_storage
from .program_types import phase
from .qpe_control import emit_controlled_effect
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def capture_qpe(owner: _Body, node: ast.Call) -> None:
    """Keep numerical phase formats separate from raw mutable helper borrowing."""
    require_root_storage(owner, node, unitary_region=True)
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
            "use a static zero-argument lambda containing one static quantum call",
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
            "use one positional quantum call in a zero-argument lambda",
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
    flush_evolutions(owner)
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
    storage = dict(owner.registers), dict(owner.ports), dict(owner.widths)
    start, effect = len(owner.operations), owner.effect
    owner.region_depth += 1
    try:
        if (
            owner.resolve_callable(body.func) is phase
            and len(body.args) == 2
            and owner.phase_number(body.args[0]) is None
            and (angle := owner.phase_number(body.args[1])) is not None
        ):
            capture_polynomial_phase(owner, body.args[0], angle, body)
        else:
            owner.call(body)
        flush_evolutions(owner)
    finally:
        owner.region_depth -= 1
    effects = tuple(owner.operations[start:])
    del owner.operations[start:]
    owner.effect = effect
    allowed = {
        "qreg.gate",
        "qreg.pair_gate",
        "qreg.phase",
        "qreg.control_phase",
        "qreg.control_x",
        "qreg.array_phase_flip",
        "qreg.pauli_rotation",
        "qreg.unitary",
        "qreg.inplace_prepare_amplitudes",
    }
    if (
        not effects
        or storage != (owner.registers, owner.ports, owner.widths)
        or any(
            op.name not in allowed
            or op.name == "qreg.phase"
            and "callee" in op.attributes
            for op in effects
        )
    ):
        _error(
            owner.source,
            body,
            "program.qpe",
            "callback requires nonempty pure quantum effects without allocation,ownership changes or Output publication",
        )
    if any(counter in op.operands[1:] for op in effects):
        _error(
            owner.source,
            body,
            "program.qpe",
            "phase counter and callback data must be disjoint",
        )

    def emit(
        name: str, values: tuple[Value, ...], attributes: dict[str, object]
    ) -> None:
        owner.emit(name, node, values, **attributes)

    for bit in range(width):
        owner.emit("qreg.gate", node, (counter,), gate="h", index=bit)
    for bit in range(width):
        if (
            len(effects) == 1
            and effects[0].name == "qreg.unitary"
            and "control_index" not in effects[0].attributes
        ):
            operation = effects[0]
            owner.emit(
                "qreg.unitary",
                node,
                (*operation.operands[1:], counter),
                **dict(operation.attributes),
                control_index=bit,
                power=1 << bit,
            )
        else:
            try:
                for _ in range(1 << bit):
                    for operation in effects:
                        emit_controlled_effect(counter, bit, operation, emit)
            except ValueError as error:
                _error(owner.source, body, "program.qpe", str(error))
    owner.emit("qreg.inverse_qft", node, (counter,))
