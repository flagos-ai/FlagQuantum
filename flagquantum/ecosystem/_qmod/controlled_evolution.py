"""Specialize pure scalar-controlled evolution using existing owned Pauli effects."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .syntax import Statement, reject

if TYPE_CHECKING:
    from .adapter import _Adapt

EVOLUTION_CALLS = frozenset(
    {
        "single_pauli_exponent",
        "suzuki_trotter",
        "multi_suzuki_trotter",
        "exponentiate",
        "qdrift",
    }
)


def controlled_evolution(owner: _Adapt, statement: Statement, index: str) -> bool:
    body = statement.control_body
    if body is None or len(statement.arguments) != 1:
        return False
    visited: set[str] = set()

    def contains(children: tuple[Statement, ...]) -> bool:
        for child in children:
            if child.token.text in EVOLUTION_CALLS:
                return True
            called = owner.callback_bindings.get(
                child.token.text
            ) or owner.definitions.get(child.token.text)
            if called is not None and called.name.text not in visited:
                visited.add(called.name.text)
                if contains(called.statements):
                    return True
            if any(
                contains(block)
                for block in (
                    child.control_body or (),
                    child.control_else or (),
                    child.inverse_body or (),
                    child.repeat_body,
                )
            ):
                return True
        return False

    if not contains((*body, *(statement.control_else or ()))):
        return False
    condition = statement.arguments[0]
    expected = 1
    source = condition
    if condition.token.text in {"==", "!="} and len(condition.children) == 2:
        source, literal = condition.children
        if literal.children or literal.token.text not in {"0", "1"}:
            reject(
                condition.token,
                "control.evolution",
                "Pauli evolution needs one scalar bit and literal zero/one",
            )
        expected = int(literal.token.text)
        if condition.token.text == "!=":
            expected = 1 - expected
    elif condition.children and condition.token.text != "[":
        reject(
            condition.token,
            "control.evolution",
            "Pauli evolution requires single-layer scalar control",
        )
    port = owner.parameters.get(source.token.text)
    if not source.children and port is not None and port.kind == "qnum":
        if owner.widths[source.token.text] != 1 or source.token.text in owner.signed:
            reject(
                source.token,
                "control.evolution",
                "numeric control requires one unsigned bit",
            )
        control = owner.array_view(source)[0]
    else:
        control = owner.gate_target(source)
    branches = [(expected, body)]
    if statement.control_else is not None:
        branches.append((1 - expected, statement.control_else))
    for value, children in branches:
        start, incoming = len(owner.operations), owner.effect
        for ordinal, child in enumerate(children):
            owner.statement(child, f"{index}.pauli{value}.{ordinal}")
        expanded = [
            (
                replace(
                    op,
                    name="qreg.pauli_rotation",
                    attributes={"support": (), "angle": 0.0},
                )
                if op.name == "qreg.gate"
                and op.attributes == {"gate": "rz", "index": 0, "angle": 0.0}
                else op
            )
            for op in owner.operations[start:]
        ]
        del owner.operations[start:]
        owner.effect = incoming
        for op in expanded:
            if op.name != "qreg.pauli_rotation" or "control_index" in op.attributes:
                reject(
                    statement.token,
                    "control.evolution",
                    "Pauli branches require only previously uncontrolled rotations; ownership changes and nested controls refuse",
                )
            if control[0] == op.operands[1]:
                reject(
                    statement.token,
                    "control.alias",
                    "control and the complete Pauli target storage must be disjoint",
                )
        if not value:
            owner.effect_op(
                "qreg.gate", statement.token, (control[0],), gate="x", index=control[1]
            )
        for op in expanded:
            owner.effect_op(
                op.name,
                statement.token,
                (*op.operands[1:], control[0]),
                **{**op.attributes, "control_index": control[1]},
            )
        if not value:
            owner.effect_op(
                "qreg.gate", statement.token, (control[0],), gate="x", index=control[1]
            )
    return True
