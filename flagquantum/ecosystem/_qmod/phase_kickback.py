"""Prove phase kickback with a local bit prepared and returned to zero."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass, replace

from .syntax import Expression, Function, Statement, Token, reject

_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_ARITHMETIC = frozenset({"+", "-", "*"})


def _plain(statement: Statement) -> bool:
    return statement == Statement(statement.token, statement.arguments)


def _atom(node: Expression, text: str) -> bool:
    return not node.children and node.token.text == text


def _unary(statement: Statement, gate: str, target: str) -> bool:
    return (
        _plain(statement)
        and statement.token.text == gate
        and len(statement.arguments) == 1
        and _atom(statement.arguments[0], target)
    )


def _prepares_minus(helper: Function, target: str) -> bool:
    return (
        len(helper.parameters) == 1
        and helper.parameters[0].role == "inout"
        and helper.parameters[0].kind == "qbit"
        and len(helper.statements) == 2
        and _unary(helper.statements[0], "X", target)
        and _unary(helper.statements[1], "H", target)
    )


def _tokens(value: object) -> Iterator[str]:
    if isinstance(value, Token):
        yield value.text
    elif isinstance(value, tuple):
        for item in value:
            yield from _tokens(item)
    elif is_dataclass(value):
        for field in fields(value):
            yield from _tokens(getattr(value, field.name))


def _predicate(
    wrapper: Function, call: Statement, helper: Function, marker: str
) -> Expression | None:
    if (
        not helper.permutation
        or helper.disable_permutation_check is not None
        or len(helper.statements) not in {1, 2}
        or len(call.arguments) != len(helper.parameters)
        or len({port.name.text for port in helper.parameters}) != len(helper.parameters)
    ):
        return None
    rewrite: dict[str, Expression] = {}
    target: str | None = None
    borrowed = {port.name.text: port for port in wrapper.parameters}
    for port, actual in zip(helper.parameters, call.arguments, strict=True):
        if (
            port.role == "inout"
            and port.kind in {"qbit", "qnum"}
            and port.width in {None, 1}
            and _atom(actual, marker)
            and target is None
        ):
            target = port.name.text
            continue
        actual_port = borrowed.get(actual.token.text)
        if (
            actual.children
            or port.role != "const"
            or port.kind not in {"qbit", "qnum"}
            or actual_port is None
            or actual_port.kind != port.kind
            or (port.width is not None and actual_port.width != port.width)
        ):
            return None
        rewrite[port.name.text] = actual
    if (
        target is None
        or len(rewrite) != len(borrowed)
        or set(borrowed) != {actual.token.text for actual in rewrite.values()}
    ):
        return None
    xor = helper.statements[0]
    if (
        not _plain(xor)
        or xor.token.text != "^="
        or len(xor.arguments) != 2
        or not _atom(xor.arguments[0], target)
    ):
        return None
    condition = xor.arguments[1]
    if condition.token.text not in _COMPARISONS or len(condition.children) != 2:
        return None

    def arithmetic(node: Expression) -> Expression | None:
        if not node.children:
            return node if node.token.text.isdecimal() else rewrite.get(node.token.text)
        if node.token.text not in _ARITHMETIC or len(node.children) != 2:
            return None
        left, right = (arithmetic(child) for child in node.children)
        return (
            None
            if left is None or right is None
            else replace(node, children=(left, right))
        )

    left, right = (arithmetic(child) for child in condition.children)
    if left is None or right is None:
        return None
    condition = replace(condition, children=(left, right))
    if not borrowed.keys() <= set(_tokens(condition)):
        return None
    if len(helper.statements) == 2:
        if not _unary(helper.statements[1], "X", target):
            return None
        condition = Expression(replace(condition.token, text="~"), (condition,))
    return condition


def _fold(
    wrapper: Function, definitions: Mapping[str, Function]
) -> tuple[Statement, str] | None:
    if (
        len(wrapper.statements) != 6
        or not wrapper.parameters
        or any(
            p.role not in {"const", "inout"} or p.kind not in {"qbit", "qnum"}
            for p in wrapper.parameters
        )
        or len({p.name.text for p in wrapper.parameters}) != len(wrapper.parameters)
    ):
        return None
    declare, allocate, prepare, mark, undo, free = wrapper.statements
    bit = declare.declaration
    if bit is None or bit.kind != "qbit" or bit.role != "local":
        return None
    name = bit.name.text
    local_names = {name, *(p.name.text for p in wrapper.parameters)}
    if (
        len(local_names) != len(wrapper.parameters) + 1
        or prepare.token.text in local_names
        or mark.token.text in local_names
        or not _plain(allocate)
        or allocate.token.text != "allocate"
        or len(allocate.arguments) not in {1, 2}
        or not _atom(allocate.arguments[-1], name)
        or (len(allocate.arguments) == 2 and not _atom(allocate.arguments[0], "1"))
        or not _unary(prepare, prepare.token.text, name)
        or not _plain(mark)
        or undo.token.text != "invert"
        or undo.inverse_body is None
        or len(undo.inverse_body) != 1
        or not _unary(undo.inverse_body[0], prepare.token.text, name)
        or undo != Statement(undo.token, (), inverse_body=undo.inverse_body)
        or not _unary(free, "free", name)
    ):
        return None
    preparation = definitions.get(prepare.token.text)
    predicate = definitions.get(mark.token.text)
    if (
        preparation is None
        or len(preparation.parameters) != 1
        or not _prepares_minus(preparation, preparation.parameters[0].name.text)
        or predicate is None
    ):
        return None
    condition = _predicate(wrapper, mark, predicate, name)
    if condition is None:
        return None
    return (
        Statement(
            replace(mark.token, text="$kickback_phase"),
            (condition, Expression(replace(mark.token, text=repr(math.pi)))),
        ),
        predicate.name.text,
    )


def expand_phase_kickbacks(definitions: Mapping[str, Function]) -> dict[str, Function]:
    """Fold an X-eigenstate kickback only after checking the complete lifecycle."""
    rewritten = dict(definitions)
    consumed = set()
    for name, function in definitions.items():
        result = _fold(function, definitions)
        if result is None:
            if function.disable_permutation_check is not None:
                reject(
                    function.disable_permutation_check,
                    "permutation",
                    "cannot prove annotated phase-kickback permutation",
                )
            continue
        phase, predicate = result
        rewritten[name] = replace(
            function, statements=(phase,), disable_permutation_check=None
        )
        consumed.add(predicate)
    if not consumed:
        return rewritten
    referenced = {
        token
        for function in rewritten.values()
        for token in _tokens(function.statements)
    }
    return {
        name: function
        for name, function in rewritten.items()
        if name not in consumed or name in referenced
    }
