"""Resolve raw borrowed numeric words onto existing physical input registers."""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from .._hybrid.model import Value
from .frontend_python import _error
from .frontend_views import locations
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def borrowed_expression(
    owner: _Body, expression: ast.expr, names: tuple[str, ...]
) -> tuple[ast.expr, tuple[str, ...], tuple[Value, ...]]:
    """Keep caller storage as graph arguments; only rewrite unsigned word reads."""
    assert owner.aliases is not None and owner.alias_ports is not None
    views = {
        name: locations(owner, ast.Name(id=name, ctx=ast.Load())) for name in names
    }
    actuals = tuple(
        dict.fromkeys(value for view in views.values() for value, _ in view)
    )
    arguments = tuple(f"__borrowed_input_{index}" for index in range(len(actuals)))
    arguments_by_value = dict(zip(actuals, arguments, strict=True))
    words: dict[str, ast.expr] = {}
    for name, view in views.items():
        port = owner.alias_ports[name]
        if port.kind not in {"qnum", "qbit"} or port.signed or port.fractional_bits:
            _error(
                owner.source,
                expression,
                "program.borrow",
                "borrowed expressions require raw unsigned numeric views",
            )
        parent = view[0][0]
        variable = ast.Name(id=arguments_by_value[parent], ctx=ast.Load())
        if view == tuple((parent, bit) for bit in range(len(view))):
            words[name] = (
                variable
                if len(view) == register_width(parent)
                else ast.BinOp(
                    left=variable,
                    op=ast.BitAnd(),
                    right=ast.Constant((1 << len(view)) - 1),
                )
            )
            continue
        word: ast.expr = ast.Constant(0)
        for bit, (value, index) in enumerate(view):
            mask = 1 << index
            selected = ast.Compare(
                left=ast.BinOp(
                    left=ast.Name(id=arguments_by_value[value], ctx=ast.Load()),
                    op=ast.BitAnd(),
                    right=ast.Constant(mask),
                ),
                ops=[ast.NotEq()],
                comparators=[ast.Constant(0)],
            )
            word = ast.BinOp(
                left=word,
                op=ast.Add(),
                right=ast.BinOp(
                    left=selected, op=ast.Mult(), right=ast.Constant(1 << bit)
                ),
            )
        words[name] = word

    class Reads(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return (
                ast.copy_location(deepcopy(words[node.id]), node)
                if node.id in words
                else node
            )

    return (
        cast(ast.expr, ast.fix_missing_locations(Reads().visit(deepcopy(expression)))),
        arguments,
        actuals,
    )
