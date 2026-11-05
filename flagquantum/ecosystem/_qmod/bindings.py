"""Lexical register-view bindings without allocating or reversing quantum data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .syntax import MAX_SCOPE_DEPTH, Statement, reject

if TYPE_CHECKING:
    from .adapter import _Adapt
    from .syntax import Parser


def is_binding_scope(statement: Statement) -> bool:
    return bool(statement.within) and all(
        child.token.text == "->" for child in statement.within or ()
    )


def parse_within(parser: Parser, allow_scopes: bool, scope_depth: int) -> Statement:
    token = parser.take("within")
    if not allow_scopes:
        reject(token, "scope", "nested scopes belong in apply, not preparation")
    if scope_depth >= MAX_SCOPE_DEPTH:
        reject(token, "budget.depth", f"scope nesting exceeds depth {MAX_SCOPE_DEPTH}")
    preparation = parser.block(
        allow_scopes=False, scope_depth=scope_depth + 1, binding_preparation=True
    )
    parser.take("apply")
    binding_only = bool(preparation) and all(s.token.text == "->" for s in preparation)
    actions = parser.block(scope_depth=scope_depth + 1, binding_actions=binding_only)
    return Statement(token, (), within=preparation, apply=actions)


def lower_binding_scope(owner: _Adapt, statement: Statement, index: str) -> bool:
    if not is_binding_scope(statement):
        return False
    views = dict(owner.views)
    widths = dict(owner.widths)
    suspended = set(owner.suspended)
    start = len(owner.operations)
    try:
        for child in statement.within or ():
            owner.bind_registers(child.token, *child.arguments)
        for ordinal, child in enumerate(statement.apply):
            owner.statement(child, f"{index}.binding{ordinal}")
        if any(
            op.name not in {"qreg.gate", "qreg.pair_gate", "qreg.control_x"}
            for op in owner.operations[start:]
        ):
            reject(
                statement.token,
                "binding.scope",
                "binding apply admits only static gates; allocation and ownership changes refuse",
            )
    finally:
        owner.views = views
        owner.widths = widths
        owner.suspended = suspended
    return True
