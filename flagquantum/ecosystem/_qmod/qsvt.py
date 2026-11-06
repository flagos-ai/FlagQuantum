"""Compose QSVT sequences from verified lexical effects."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ...compiler._construction.register_profile import register_width
from .callback_arrays import capture_effects
from .masked_selection import emit_selected
from .quantum_callbacks import Lambda
from .syntax import _COMPARISONS, Expression, Statement, Token, reject
from .unitary_effects import emit_inverse, inverse_effects, preserves_basis_labels

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Operation
    from .adapter import _Adapt
    from .phase import Location


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


def _effect_locations(op: Operation) -> set[Location]:
    operands = op.operands[1:]
    if op.name == "qreg.gate":
        return {(operands[0], op.attributes["index"])}
    if "indices" in op.attributes:
        return set(zip(operands, op.attributes["indices"], strict=True))
    return {(value, bit) for value in operands for bit in range(register_width(value))}


def lower_qsvt_lcu(owner: _Adapt, token: Token, args: tuple[Expression, ...]) -> None:
    """Share the reversible block between bounded even and odd sequences."""
    if len(args) != 7:
        reject(
            token,
            "qsvt.signature",
            "LCU needs two phase sequences, three callbacks and two bits",
        )
    even_arg, odd_arg, first, second, block, aux_arg, select_arg = args
    even, odd = owner.real_array(even_arg), owner.real_array(odd_arg)
    if (
        not 1 <= len(even) <= 255
        or len(even) % 2 != 1
        or not 2 <= len(odd) <= 256
        or len(odd) % 2 != 0
        or abs(len(even) - len(odd)) != 1
    ):
        reject(
            token,
            "qsvt.phases",
            "LCU requires adjacent even and odd polynomial degrees",
        )
    if not isinstance(first, Lambda) or not isinstance(second, Lambda):
        reject(token, "qsvt.projector", "LCU requires two lexical projector callbacks")
    if not isinstance(block, Lambda) or block.parameters or not block.body:
        reject(block.token, "qsvt.block", "LCU requires a nonempty zero-argument block")
    aux, select = owner.gate_target(aux_arg), owner.gate_target(select_arg)
    if aux == select:
        reject(token, "qsvt.alias", "LCU selector and auxiliary must be disjoint")

    def capture_projector(projector: Lambda) -> tuple[Operation, ...]:
        effects = capture_effects(
            owner, projector.token, (_projector_body(owner, projector, aux_arg),)
        )
        support = {location for op in effects for location in _effect_locations(op)}
        if select in support or any(
            not preserves_basis_labels(op, support - {aux}) for op in effects
        ):
            reject(
                projector.token,
                "qsvt.projector",
                "projector may write only its auxiliary and cannot read the selector",
            )
        return effects

    left, right = capture_projector(first), capture_projector(second)
    forward = capture_effects(owner, block.token, block.body)
    if any({aux, select} & _effect_locations(op) for op in forward):
        reject(
            block.token, "qsvt.alias", "block cannot use the LCU selector or auxiliary"
        )
    backward = inverse_effects(token, forward)
    left_inverse, right_inverse = inverse_effects(token, left), inverse_effects(
        token, right
    )

    def replay(effects: tuple[Operation, ...]) -> None:
        for op in effects:
            owner.effect_op(op.name, token, op.operands[1:], **op.attributes)

    def phase(index: int) -> None:
        effects, undo = (right, right_inverse) if index % 2 else (left, left_inverse)
        replay(effects)
        for branch, angles in enumerate((even, odd)):
            if index >= len(angles):
                continue
            angle = angles[index]
            # The branch phase is part of controlled RZ and survives interference.
            owner.effect_op(
                "qreg.control_phase",
                token,
                (select[0],),
                indices=(select[1],),
                control_state=branch,
                angle=-angle / 2,
            )
            owner.effect_op(
                "qreg.control_phase",
                token,
                (select[0], aux[0]),
                indices=(select[1], aux[1]),
                control_state=branch | 2,
                angle=angle,
            )
        replay(undo)

    owner.effect_op("qreg.gate", token, (aux[0],), gate="h", index=aux[1])
    phase(0)
    common = min(len(even), len(odd))
    for index in range(1, common):
        replay(forward if index % 2 else backward)
        phase(index)
    emit_selected(
        owner,
        token,
        (select,),
        int(len(odd) > len(even)),
        forward if common % 2 else backward,
    )
    phase(common)
    owner.effect_op("qreg.gate", token, (aux[0],), gate="h", index=aux[1])
