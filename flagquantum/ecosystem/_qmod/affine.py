"""Bounded affine updates and non-escaping predicate snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from operator import eq, ge, gt, le, lt, ne
from typing import TYPE_CHECKING

from .syntax import Expression, Statement, Token, reject

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Value
    from .adapter import _Adapt

_COMPARE = {"<": lt, "<=": le, ">": gt, ">=": ge, "==": eq, "!=": ne}
_MAX_BITS = 8


def affine_bits(
    owner: _Adapt, expression: Expression
) -> tuple[int, dict[tuple[Value, int], int]]:
    """Map an integer sum/difference to classical offset and unsigned bit weights."""
    weights: dict[tuple[Value, int], int] = {}
    touched: set[tuple[Value, int]] = set()

    def collect(node: Expression, sign: int) -> int:
        if node.token.text in {"+", "-"} and node.children:
            left, *right = node.children
            if not right:
                return collect(left, -sign)
            return collect(left, sign) + collect(
                right[0], sign if node.token.text == "+" else -sign
            )
        port = owner.parameters.get(node.token.text)
        if not node.children and port is not None and port.kind == "qnum":
            view = owner.array_view(node)
            touched.update(view)
            if len(touched) > _MAX_BITS:
                reject(
                    node.token,
                    "budget",
                    "affine arithmetic supports at most eight source bits",
                )
            for bit, location in enumerate(view):
                weights[location] = weights.get(location, 0) + sign * (1 << bit)
            return 0
        return sign * owner.static_integer(node)

    offset = collect(expression, 1)
    if abs(offset) > 256:
        reject(
            expression.token,
            "arithmetic",
            "affine classical offset must be in -256..256",
        )
    return offset, weights


def _add_power(
    owner: _Adapt,
    token: Token,
    target: Sequence[tuple[Value, int]],
    start: int,
    negative: bool,
    control: tuple[Value, int] | None = None,
) -> None:
    order = range(start, len(target))
    controls = () if control is None else (control,)
    for bit in order if negative else reversed(order):
        locations = (*controls, *target[start:bit], target[bit])
        if len(locations) > 1:
            owner.emit_indexed_x(token, locations)
        else:
            value, wire = target[bit]
            owner.effect_op("qreg.gate", token, (value,), gate="x", index=wire)


def lower_affine_update(
    owner: _Adapt, token: Token, target: Expression, amount: Expression
) -> None:
    view = owner.array_view(target)
    if len(view) > _MAX_BITS:
        reject(
            target.token, "budget", "affine update supports at most eight target bits"
        )
    offset, weights = affine_bits(owner, amount)
    if set(view).intersection(weights):
        reject(
            target.token, "alias", "affine update source and target must be disjoint"
        )
    for control, weight in [(None, offset), *weights.items()]:
        for bit in range(len(view)):
            if abs(weight) & (1 << bit):
                _add_power(owner, token, view, bit, weight < 0, control)
    for value, wire in (*weights, view[0]):
        owner.effect_op("qreg.gate", token, (value,), gate="i", index=wire)


def snapshot_start(statements: tuple[Statement, ...]) -> int | None:
    """Recognize a fresh local followed by one update and terminal flag reads."""
    for index, statement in enumerate(statements[:-3]):
        local = statement.declaration
        if local is None or local.kind != "qnum" or local.width is not None:
            continue
        assignment, _, *guards = statements[index + 1 :]
        if (
            assignment.token.text != "="
            or assignment.arguments[0].token.text != local.name.text
        ):
            continue
        if all(
            guard.token.text == "control"
            and guard.control_else is None
            and len(guard.arguments) == 1
            and guard.arguments[0].token.text in _COMPARE
            and len(guard.arguments[0].children) == 2
            and not guard.arguments[0].children[0].children
            and guard.arguments[0].children[0].token.text == local.name.text
            and guard.control_body is not None
            and len(guard.control_body) == 1
            and guard.control_body[0].token.text == "X"
            and len(guard.control_body[0].arguments) == 1
            for guard in guards
        ):
            return index
    return None


def lower_snapshot(
    owner: _Adapt, statements: tuple[Statement, ...], index: int
) -> None:
    declaration, assignment, update, *guards = statements[index:]
    local = declaration.declaration
    assert local is not None
    if local.name.text in owner.parameters:
        reject(local.name, "name", "snapshot requires a fresh local")
    offset, weights = affine_bits(owner, assignment.arguments[1])
    if not weights:
        reject(assignment.token, "arithmetic", "snapshot requires quantum inputs")
    destinations = []
    for guard in guards:
        assert guard.control_body is not None
        flag = owner.gate_target(guard.control_body[0].arguments[0], allow_numeric=True)
        if flag in weights:
            reject(guard.token, "alias", "snapshot inputs and flag must be disjoint")
        destinations.append(flag)
    flags = {value for value, _ in destinations}
    if update.token.text != "+=" and update.token.text not in owner.definitions:
        reject(
            update.token,
            "scope",
            "snapshot requires an integer update or a checked helper",
        )
    incoming, start = owner.effect, len(owner.operations)
    owner.statement(update, str(index + 2))
    effects = owner.operations[start:]
    if any(
        flags.intersection(op.operands[1:])
        or not (
            op.name == "qreg.control_x"
            or (op.name == "qreg.gate" and op.attributes.get("gate") in {"i", "x"})
        )
        for op in effects
    ):
        reject(
            update.token,
            "scope",
            "snapshot update must be a disjoint reversible integer update",
        )
    del owner.operations[start:]
    owner.effect = incoming

    # These terminal flag operations depend on the captured inputs and commute
    # with the disjoint update. Emit them while those inputs still hold that value.
    wires = tuple(weights)
    for value, wire in (*wires, *destinations):
        owner.effect_op("qreg.gate", assignment.token, (value,), gate="i", index=wire)
    for guard, flag in zip(guards, destinations, strict=True):
        condition = guard.arguments[0]
        threshold = owner.static_integer(condition.children[1])
        for word in range(1 << len(wires)):
            captured = offset + sum(
                weight
                for bit, weight in enumerate(weights.values())
                if word & (1 << bit)
            )
            if not _COMPARE[condition.token.text](captured, threshold):
                continue
            zeros = [
                location for bit, location in enumerate(wires) if not word & (1 << bit)
            ]
            for register, wire in zeros:
                owner.effect_op(
                    "qreg.gate", guard.token, (register,), gate="x", index=wire
                )
            owner.emit_indexed_x(guard.token, (*wires, flag))
            for register, wire in reversed(zeros):
                owner.effect_op(
                    "qreg.gate", guard.token, (register,), gate="x", index=wire
                )
    for op in effects:
        owner.effect_op(op.name, update.token, op.operands[1:], **op.attributes)
