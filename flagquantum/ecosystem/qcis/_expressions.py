"""Restricted angle syntax lowered to existing Core parameter expressions."""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from typing import TypeAlias

from ...core.parameters import Parameter, ParameterExpression

Angle: TypeAlias = float | Parameter | ParameterExpression
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z", re.ASCII)


def _normalize_names(source: str) -> tuple[str, dict[str, str]]:
    tokens: list[tuple[int, str]] = []
    names: dict[str, str] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            text = token.string
            if token.type == tokenize.NAME:
                if not _NAME.fullmatch(text) or text in {"True", "False", "None"}:
                    raise ValueError("invalid parameter name")
                # QCIS names may be Python keywords: shield them before
                # AST parsing, then restore the original names in Core parameters.
                alias = f"_qcis_parameter_{len(names)}"
                names[alias] = text
                text = alias
            tokens.append((token.type, text))
    except (tokenize.TokenError, IndentationError) as exc:
        raise ValueError("malformed angle expression") from exc
    return tokenize.untokenize(tokens), names


def parse_angle(source: str) -> Angle:
    """Parse without executing code; reject syntax outside Core's algebra."""
    if len(source) > 4096:
        raise ValueError("angle expression exceeds 4096 characters")
    # Preserve decimal spellings (including leading zeroes) accepted before
    # symbolic expressions were supported; Python's integer grammar differs.
    if _DECIMAL.fullmatch(source):
        value = float(source)
        if not math.isfinite(value):
            raise ValueError("the angle must be finite")
        return value
    normalized, names = _normalize_names(source)
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("malformed angle expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("angle expression exceeds 256 syntax nodes")

    def visit(node: ast.expr, depth: int = 0) -> Angle:
        if depth > 32:
            raise ValueError("angle expression exceeds 32 nesting levels")
        value: Angle
        if isinstance(node, ast.Constant):
            literal = ast.get_source_segment(normalized, node) or ""
            if not _DECIMAL.fullmatch(literal):
                raise ValueError("expected a decimal real number")
            value = float(literal)
        elif isinstance(node, ast.Name):
            name = names[node.id]
            if name.lower() in {"nan", "inf", "infinity"}:
                raise ValueError("invalid parameter name or non-finite angle")
            value = math.pi if name == "pi" else Parameter(name)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            operand = visit(node.operand, depth + 1)
            value = -operand if isinstance(node.op, ast.USub) else operand
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = visit(node.left, depth + 1)
            right = visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                if not isinstance(right, float) or right == 0:
                    raise ValueError("division requires a nonzero constant denominator")
                if isinstance(left, float):
                    value = left / right
                else:
                    reciprocal = 1.0 / right
                    if not math.isfinite(reciprocal):
                        raise ValueError("division produces a non-finite coefficient")
                    value = left * reciprocal
        else:
            raise ValueError(
                "only decimal numbers, names, pi, +, -, *, / and parentheses are supported"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("the angle or a constant subexpression must be finite")
        return value

    return visit(tree.body)
