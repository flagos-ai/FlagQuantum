"""One strict decimal scanner shared by existing literal consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .syntax import Expression, Parser


def parse_literal(parser: Parser) -> Expression:
    # Imported only when the fully initialized source parser reads a literal.
    from .syntax import Expression, Token, reject

    integer = parser.current
    if not integer.text.isdecimal() or parser.tokens[parser.position + 1].text != ".":
        return parser.expression(allow_slice=True)
    parser.position += 1
    dot = parser.take(".")
    fraction = parser.current
    if (
        not fraction.text.isdecimal()
        or dot.location.line != integer.location.line
        or dot.location.column != integer.location.column + len(integer.text)
        or fraction.location.line != dot.location.line
        or fraction.location.column != dot.location.column + 1
    ):
        reject(
            dot,
            "angle",
            "decimal angles require contiguous digits on both sides of the point",
        )
    text = integer.text + "." + fraction.text
    if len(text) > 78:
        reject(integer, "budget.number", "angle literal exceeds 78 characters")
    parser.position += 1
    return Expression(Token(text, integer.location))
