"""Parse the bounded Qmod expression grammar."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from .angles import SCIENTIFIC_LITERAL
from .syntax_model import (
    _COMPARISONS,
    EXTREMA,
    MAX_EXPRESSION_DEPTH,
    Expression,
    Token,
    reject,
)

if TYPE_CHECKING:
    from .syntax import Parser


def expression(
    self: Parser,
    minimum: int = 0,
    nesting: int = 0,
    *,
    static: bool = False,
    decimal: bool = False,
    difference: bool = False,
    allow_slice: bool = False,
    real: bool = False,
    depth_limit: int = MAX_EXPRESSION_DEPTH,
) -> Expression:
    token = self.current
    if nesting >= depth_limit:
        reject(
            token,
            "budget.depth",
            f"expression nesting exceeds {depth_limit}",
        )
    if (not static and token.text in {"~", "not"}) or (
        (static or difference) and token.text == "-"
    ):
        self.take(token.text)
        operand = self.expression(
            4 if token.text == "not" else 25,
            nesting + 1,
            static=static,
            decimal=decimal,
            difference=difference,
            real=real,
            depth_limit=depth_limit,
        )
        if operand.depth >= depth_limit:
            reject(
                token,
                "budget.depth",
                f"expression tree exceeds depth {depth_limit}",
            )
        if token.text == "not":
            zero = Expression(Token("0", token.location))
            left = Expression(
                Token("==", token.location), (operand, zero), operand.depth + 1
            )
        else:
            left = Expression(token, (operand,), operand.depth + 1)
    elif token.text == "(":
        self.take("(")
        left = self.expression(
            nesting=nesting + 1,
            static=static,
            decimal=decimal,
            difference=difference,
            real=real,
            depth_limit=depth_limit,
        )
        self.take(")")
    elif (
        (static or real)
        and token.text == "log"
        and self.tokens[self.position + 1].text == "("
    ):
        self.take("log")
        self.take("(")
        argument = self.expression(
            static=True, nesting=nesting + 1, depth_limit=depth_limit
        )
        self.take(",")
        self.take("2")
        self.take(")")
        left = Expression(
            replace(token, text="$exact_log2"), (argument,), argument.depth + 1
        )
    elif static and token.text == "ceiling":
        left = self.binary_log_width(
            symbolic=True, nesting=nesting, depth_limit=depth_limit
        )
    elif (
        (static or real)
        and token.text == "floor"
        and self.tokens[self.position + 1].text == "("
    ):
        self.take("floor")
        self.take("(")
        argument = self.expression(
            nesting=nesting + 1,
            static=static,
            real=real,
            decimal=decimal,
            difference=difference,
            depth_limit=depth_limit,
        )
        self.take(")")
        left = Expression(token, (argument,), argument.depth + 1)
    elif (
        real
        and token.text in {"acos", "asin", "sqrt", "exp"}
        and self.tokens[self.position + 1].text == "("
    ):
        self.take(token.text)
        self.take("(")
        argument = self.expression(
            nesting=nesting + 1,
            decimal=True,
            difference=True,
            real=True,
            depth_limit=depth_limit,
        )
        self.take(")")
        left = Expression(token, (argument,), argument.depth + 1)
    elif (
        token.text in {"qft_const_adder_phase", "mod_inverse"}
        and self.tokens[self.position + 1].text == "("
    ):
        self.take(token.text)
        self.take("(")
        arguments = []
        for argument_index in range(3 if token.text == "qft_const_adder_phase" else 2):
            if argument_index:
                self.take(",")
            arguments.append(
                self.expression(
                    static=True, nesting=nesting + 1, depth_limit=depth_limit
                )
            )
        self.take(")")
        left = Expression(
            token, tuple(arguments), 1 + max(arg.depth for arg in arguments)
        )
    elif decimal and SCIENTIFIC_LITERAL.fullmatch(token.text):
        left = self.angle_argument()
    elif token.text.isdecimal():
        if decimal and self.tokens[self.position + 1].text == ".":
            left = self.angle_argument()
        else:
            self.position += 1
            left = Expression(token)
    else:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", token.text):
            reject(
                token,
                "expression",
                "expected a nonnegative integer, output name or parentheses",
            )
        left = Expression(self.name())
    if (
        not left.children
        and token.text in EXTREMA
        and not static
        and self.current.text == "("
    ):
        self.take("(")
        first = self.expression(
            nesting=nesting + 1,
            decimal=True,
            difference=difference,
            real=real,
            depth_limit=depth_limit,
        )
        self.take(",")
        second = self.expression(
            nesting=nesting + 1,
            decimal=True,
            difference=difference,
            real=real,
            depth_limit=depth_limit,
        )
        self.take(")")
        depth = max(first.depth, second.depth) + 1
        if depth > depth_limit:
            reject(
                token,
                "budget.depth",
                f"expression tree exceeds depth {depth_limit}",
            )
        left = Expression(token, (first, second), depth)
    while True:
        if not left.children and left.token.text.isdecimal():
            break
        if (
            static
            and self.current.text == ":"
            and [
                item.text for item in self.tokens[self.position + 1 : self.position + 3]
            ]
            == [":", "len"]
        ):
            separator = self.take(":")
            self.take(":")
            length_token = self.take("len")
            left = Expression(
                Token(".", separator.location),
                (left, Expression(length_token)),
                left.depth + 1,
            )
        elif self.current.text == "." and (
            static
            or self.records
            or (
                real
                and self.tokens[self.position + 1].text
                in {"len", "size", "index", "pauli", "fraction_digits"}
            )
        ):
            dot = self.take(".")
            member = Expression(self.name())
            left = Expression(dot, (left, member), left.depth + 1)
        elif self.current.text == "[" and (
            (not left.children)
            or (not static and left.token.text == ":")
            or (real and left.token.text == "[")
            or (
                self.records
                and (
                    left.token.text == "."
                    or (left.token.text == "[" and left.children[0].token.text == ".")
                )
            )
        ):
            bracket = self.take("[")
            index = self.expression(
                nesting=nesting + 1, static=True, depth_limit=depth_limit
            )
            if allow_slice and self.current.text == ":":
                colon = self.take(":")
                stop = self.expression(
                    nesting=nesting + 1, static=True, depth_limit=depth_limit
                )
                left = Expression(
                    colon,
                    (left, index, stop),
                    max(left.depth, index.depth, stop.depth) + 1,
                )
            else:
                left = Expression(
                    bracket, (left, index), max(left.depth, index.depth) + 1
                )
            self.take("]")
        else:
            break
        if left.depth > depth_limit:
            reject(
                token,
                "budget.depth",
                f"expression tree exceeds depth {depth_limit}",
            )
    precedence = {
        "or": 2,
        "and": 3,
        **dict.fromkeys(_COMPARISONS, 5),
        "|": 6,
        "^": 7,
        "&": 8,
        "+": 10,
        "*": 20,
        "%": 20,
        "**": 30,
    }
    if static:
        precedence = {"+": 10, "-": 10, "*": 20, "/": 20, "%": 20, "**": 30}
    elif difference:
        precedence["-"] = 10
    if real:
        precedence["/"] = 20
    compared = False
    while precedence.get(self.current.text, -1) >= minimum:
        operator = self.current
        if operator.text in _COMPARISONS:
            if compared:
                reject(
                    operator,
                    "operator",
                    "comparison chains require explicit parentheses",
                )
            compared = True
        self.position += 1
        level = precedence[operator.text]
        right = self.expression(
            level if operator.text == "**" else level + 1,
            nesting + 1,
            static=static,
            decimal=decimal,
            difference=difference,
            real=real,
            depth_limit=depth_limit,
        )
        if (
            operator.text == "**"
            and not (static or real)
            and (
                right.children
                or not right.token.text.isdecimal()
                or not 1 <= int(right.token.text) <= 8
            )
        ):
            reject(
                operator,
                "operator",
                "power requires a literal exponent in 1..8",
            )
        depth = max(left.depth, right.depth) + 1
        if depth > depth_limit:
            reject(
                operator,
                "budget.depth",
                f"expression tree exceeds depth {depth_limit}",
            )
        left = Expression(operator, (left, right), depth)
    if self.current.text in {"/", "[", ".", "("} or (
        self.current.text == "-" and not (static or difference)
    ):
        reject(
            self.current,
            "operator",
            "this operator, path or numeric form is not admitted",
        )
    return left
