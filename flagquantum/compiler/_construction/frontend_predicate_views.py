"""Normalize unsigned predicate views without copying quantum storage."""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING

from .._phase_polynomial import Location
from .frontend_python import _error
from .frontend_views import locations
from .register_profile import register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def has_predicate_views(owner: _Body, expression: ast.expr) -> bool:
    """Leave existing array-element and classical-index capture paths intact."""
    ports = owner.alias_ports if owner.aliases is not None else owner.ports
    assert ports is not None
    for node in ast.walk(expression):
        if not isinstance(node, ast.Subscript):
            continue
        base = node.value
        while isinstance(base, ast.Subscript):
            base = base.value
        if (
            isinstance(base, ast.Name)
            and base.id in ports
            and ports[base.id].kind == "qnum"
        ):
            return True
    return False


def predicate_views(owner: _Body, expression: ast.expr) -> tuple[ast.expr, int]:
    """Express little-endian bit views using the existing unsigned word graph.

    Extraction is Boolean arithmetic on the borrowed input, not a fresh data
    register. Read accounting uses unique physical bits, including cancelled
    terms. Existing register-level borrowing still forbids mutating any source
    register while its predicate is live.
    """
    reads: set[Location] = set()
    visible = owner.alias_ports if owner.aliases is not None else owner.ports
    assert visible is not None
    ports = dict(visible)

    class Views(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
            base = node.value
            while isinstance(base, ast.Subscript):
                base = base.value
            if not isinstance(base, ast.Name) or base.id not in ports:
                _error(owner.source, node, "program.view", "use a quantum bit view")
            if ports[base.id].kind != "qnum":
                _error(
                    owner.source,
                    node,
                    "program.view",
                    "numeric predicate slices require an unsigned integer QNum",
                )
            whole = locations(owner, base)
            value = whole[0][0]
            if whole != tuple((value, bit) for bit in range(register_width(value))):
                _error(
                    owner.source,
                    node,
                    "program.borrow",
                    "predicate views must borrow a complete raw parent register",
                )
            selected = locations(owner, node)
            reads.update(selected)
            result: ast.expr = ast.Constant(value=0)
            for output_bit, (_, input_bit) in enumerate(selected):
                mask = 1 << input_bit
                bit = ast.Compare(
                    left=ast.BinOp(
                        left=deepcopy(base), op=ast.BitAnd(), right=ast.Constant(mask)
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(mask)],
                )
                term = ast.BinOp(
                    left=bit, op=ast.Mult(), right=ast.Constant(1 << output_bit)
                )
                result = ast.BinOp(left=result, op=ast.Add(), right=term)
            return ast.copy_location(ast.fix_missing_locations(result), node)

        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id in ports:
                reads.update(locations(owner, node))
            return node

    normalized = Views().visit(deepcopy(expression))
    assert isinstance(normalized, ast.expr)
    if len(reads) > 8:
        _error(
            owner.source,
            expression,
            "program.control",
            "control predicate exceeds eight input bits",
        )
    return normalized, len(reads)
