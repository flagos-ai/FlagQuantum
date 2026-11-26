"""Capture quantum statements, observables and terminal results."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from flagquantum._import_compat import alias_module as _alias_module
from flagquantum.compiler.frontends._structured.frontend import expressions
from flagquantum.compiler.frontends._structured.frontend.expressions import _call_name
from flagquantum.compiler.frontends._structured.model import (
    BOOL,
    QUANTUM_EFFECT,
    Operation,
    scalar_type,
)

if TYPE_CHECKING:
    from flagquantum.compiler.frontends._structured.frontend.program import _Capture


def capture_quantum_call(owner: _Capture, call: ast.Call) -> None:
    name = _call_name(call).replace("_", "").lower()
    if name == "angleembedding":
        if len(call.args) != 1:
            owner.fail(call, "quantum.arguments", "AngleEmbedding requires data")
        data = expressions.emit_expression(owner, call.args[0])
        wires = expressions.static_wires(owner, call)
        owner.effect = owner.emit(
            call,
            "quantum.angle_embedding",
            operands=(data, owner.require_effect(call)),
            result_types=(QUANTUM_EFFECT,),
            attributes={"wires": wires},
        )[0]
        return
    if name in {"rx", "ry"}:
        if len(call.args) != 1:
            owner.fail(
                call, "quantum.arguments", f"{name.upper()} requires one parameter"
            )
        parameter = expressions.emit_expression(owner, call.args[0])
        wire = expressions.dynamic_wires(owner, call, expected=1)[0]
        owner.effect = owner.emit(
            call,
            f"quantum.{name}",
            operands=(parameter, wire, owner.require_effect(call)),
            result_types=(QUANTUM_EFFECT,),
        )[0]
        return
    if name in {"cx", "cnot"}:
        control, target = expressions.dynamic_wires(owner, call, expected=2)
        owner.effect = owner.emit(
            call,
            "quantum.cx",
            operands=(control, target, owner.require_effect(call)),
            result_types=(QUANTUM_EFFECT,),
        )[0]
        return
    if name in {"h", "x"}:
        if call.args:
            owner.fail(
                call,
                "quantum.arguments",
                f"{name.upper()} accepts only the wires keyword",
            )
        wire = expressions.dynamic_wires(owner, call, expected=1)[0]
        owner.effect = owner.emit(
            call,
            f"quantum.{name}",
            operands=(wire, owner.require_effect(call)),
            result_types=(QUANTUM_EFFECT,),
        )[0]
        return
    if name == "reset":
        if call.args:
            owner.fail(
                call,
                "quantum.arguments",
                "reset accepts only the wires keyword",
            )
        wire = expressions.dynamic_wires(owner, call, expected=1)[0]
        owner.effect = owner.emit(
            call,
            "quantum.reset",
            operands=(wire, owner.require_effect(call)),
            result_types=(QUANTUM_EFFECT,),
        )[0]
        return
    if name == "measure":
        owner.fail(
            call,
            "quantum.measurement_value",
            "measure must be assigned to a local name",
        )
    owner.fail(
        call,
        "quantum.unsupported",
        f"unsupported quantum call {_call_name(call)!r}",
    )


def capture_return(owner: _Capture, statement: ast.Return) -> None:
    if isinstance(statement.value, ast.Name):
        result = expressions.emit_expression(owner, statement.value)
        if result.type != BOOL:
            owner.fail(
                statement.value,
                "function.return",
                "a direct return is limited to a measurement bool",
            )
        effect = owner.require_effect(statement)
        owner.effect = None
        owner.operations.append(
            Operation(
                "program.return",
                operands=(result, effect),
                location=owner.location(statement),
            )
        )
        return
    if statement.value is None or not isinstance(statement.value, ast.Call):
        owner.fail(
            statement,
            "function.return",
            "return must contain an expectation or measurement bool",
        )
    call = statement.value
    name = _call_name(call).replace("_", "").lower()
    if name not in {"expval", "expectation"} or len(call.args) != 1 or call.keywords:
        owner.fail(
            call,
            "quantum.expectation",
            "return must call expval or expectation with one observable",
        )
    terms = parse_observable(owner, call.args[0])
    result = owner.emit(
        call,
        "quantum.expectation",
        operands=(owner.require_effect(call),),
        result_types=(scalar_type("float64"),),
        attributes={"terms": terms},
    )[0]
    owner.effect = None
    owner.operations.append(
        Operation(
            "program.return",
            operands=(result,),
            location=owner.location(statement),
        )
    )


def parse_observable(owner: _Capture, node: ast.expr) -> tuple[tuple[str, int], ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return parse_observable(owner, node.left) + parse_observable(owner, node.right)
    if isinstance(node, ast.Call):
        name = _call_name(node).replace("_", "").lower()
        paulis = {"paulix": "x", "pauliy": "y", "pauliz": "z"}
        if name in paulis and len(node.args) == 1 and not node.keywords:
            return (
                (
                    paulis[name],
                    expressions.literal_non_negative_int(
                        owner, node.args[0], "quantum.observable_wire"
                    ),
                ),
            )
    try:
        literal = ast.literal_eval(node)
        terms = tuple((str(name).lower(), int(wire)) for name, wire in literal)
    except (TypeError, ValueError):
        owner.fail(node, "quantum.observable", "observable expression is unsupported")
    if not terms or any(
        name not in {"x", "y", "z"} or wire < 0 for name, wire in terms
    ):
        owner.fail(node, "quantum.observable", "observable terms are invalid")
    return terms


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (
    capture_quantum_call,
    capture_return,
    parse_observable,
):
    _identity_export.__module__ = "flagquantum.compiler._hybrid.frontend.quantum"
del _identity_export

_alias_module("flagquantum.compiler._hybrid.frontend.quantum", __name__)
