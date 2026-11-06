"""Compose the odd-degree QSVT sequence from verified lexical effects."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .callback_arrays import capture_effects
from .quantum_callbacks import Lambda
from .syntax import _COMPARISONS, Expression, Statement, Token, reject
from .unitary_effects import emit_inverse

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Operation
    from .adapter import _Adapt


def _projector_body(
    owner: _Adapt, projector: Lambda, auxiliary: Expression
) -> Statement:
    """Prove the projector body is exactly a comparison XOR into its marker."""
    if len(projector.parameters) != 1 or len(projector.body) != 1:
        reject(
            projector.token,
            "qsvt.projector",
            "projector needs one marker and one action",
        )
    marker = projector.parameters[0].text
    statement = projector.body[0]
    helper = owner.definitions.get(statement.token.text)
    checked = statement
    flag = marker
    if helper is not None:
        positions = [
            i
            for i, arg in enumerate(statement.arguments)
            if not arg.children and arg.token.text == marker
        ]
        if (
            len(statement.arguments) != len(helper.parameters)
            or len(positions) != 1
            or len(helper.statements) != 1
            or any(port.role != "inout" for port in helper.parameters)
            or helper.parameters[positions[0]].kind != "qbit"
        ):
            reject(
                statement.token,
                "qsvt.projector",
                "projector must forward one writable marker",
            )
        checked = helper.statements[0]
        flag = helper.parameters[positions[0]].name.text
    if (
        checked.token.text != "^="
        or len(checked.arguments) != 2
        or checked.arguments[0].children
        or checked.arguments[0].token.text != flag
        or checked.arguments[1].token.text not in _COMPARISONS
    ):
        reject(
            checked.token,
            "qsvt.projector",
            "projector may only XOR a comparison into its marker",
        )

    def bind(node: Expression) -> Expression:
        if not node.children:
            return auxiliary if node.token.text == marker else node
        if node.token.text == ".":
            root, field = node.children
            return replace(node, children=(bind(root), field))
        return replace(node, children=tuple(bind(child) for child in node.children))

    return replace(statement, arguments=tuple(bind(arg) for arg in statement.arguments))


def lower_qsvt_inversion(
    owner: _Adapt, token: Token, args: tuple[Expression, ...]
) -> None:
    if len(args) != 4:
        reject(
            token,
            "signature",
            "qsvt_inversion requires phases, projector, block encoding and auxiliary bit",
        )
    phases_arg, projector, block, auxiliary = args
    phases = owner.real_array(phases_arg)
    if not 2 <= len(phases) <= 256 or len(phases) % 2:
        reject(
            phases_arg.token,
            "qsvt.phases",
            "inversion needs an even phase count in 2..256",
        )
    if not isinstance(projector, Lambda):
        reject(
            projector.token, "qsvt.projector", "expected a lexical projector callback"
        )
    if not isinstance(block, Lambda) or block.parameters or not block.body:
        reject(
            block.token,
            "qsvt.block",
            "expected a nonempty zero-argument block callback",
        )
    value, bit = owner.gate_target(auxiliary)
    projector_effects = capture_effects(
        owner, projector.token, (_projector_body(owner, projector, auxiliary),)
    )
    unitary = capture_effects(owner, block.token, block.body)
    if any(value in op.operands[1:] for op in unitary):
        reject(
            block.token,
            "qsvt.alias",
            "block encoding and auxiliary storage must be disjoint",
        )

    def replay(effects: tuple[Operation, ...]) -> None:
        for op in effects:
            owner.effect_op(op.name, token, op.operands[1:], **op.attributes)

    owner.effect_op("qreg.gate", token, (value,), gate="h", index=bit)
    for i, angle in enumerate(phases):
        # An odd singular-value transform preserves the orientation of its block.
        # Inversion exchanges left and right singular vectors, so start with U†.
        if i % 2:
            emit_inverse(owner, token, unitary)
        elif i:
            replay(unitary)
        replay(projector_effects)
        owner.effect_op("qreg.gate", token, (value,), gate="rz", index=bit, angle=angle)
        emit_inverse(owner, token, projector_effects)
    owner.effect_op("qreg.gate", token, (value,), gate="h", index=bit)
