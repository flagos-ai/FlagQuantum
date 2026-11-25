"""Qmod tokens, bounded syntax records and source diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .amplification import Amplification

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Literal, NoReturn

from ...compiler._hybrid.model import SourceLocation
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from .angles import SCIENTIFIC_LITERAL

MAX_BYTES = 524288

MAX_TOKENS = 65536

MAX_DEPTH = 40

MAX_EXPRESSION_DEPTH = 128

MAX_PHASE_EXPRESSION_DEPTH = 256

MAX_SCOPE_DEPTH = 32

MAX_FUNCTIONS = 64

ANGLE_CALLS = frozenset(
    {"PHASE", "RX", "RY", "RZ", "RZZ", "CRX", "CRY", "CPHASE", "R", "U"}
)

_TOKEN = re.compile(
    r"(?P<space>\s+)|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)"
    rf"|(?P<scientific>{SCIENTIFIC_LITERAL.pattern})(?=j?(?:[^A-Za-z_0-9]|$))"
    r"|(?P<number>[0-9]+)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<symbol>\*\*|\^=|\+=|->|==|!=|<=|>=|[{}()<>,:;=+*%\[\].^/\-&|~@])"
)

_RESERVED = frozenset(
    {
        "qfunc",
        "qstruct",
        "output",
        "input",
        "const",
        "qbit",
        "qnum",
        "not",
        "true",
        "false",
    }
)

PREPARATIONS = {
    "prepare_amplitudes",
    "inplace_prepare_amplitudes",
    "prepare_state",
    "inplace_prepare_state",
}

EXTREMA = frozenset({"min", "max"})

_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})


@dataclass(frozen=True)
class Token:
    text: str
    location: SourceLocation


def reject(token: Token, code: str, message: str) -> NoReturn:
    """Use existing diagnostic types, retaining the native source coordinate."""
    loc = token.location
    raise HybridVerificationError(
        (
            Diagnostic(
                f"qmod.{code}", f"{loc.filename}:{loc.line}:{loc.column}: {message}"
            ),
        )
    )


def exact_unsigned_literal(text: str) -> int | None:
    whole, _, fraction = text.partition(".")
    if whole.isdecimal() and all(digit == "0" for digit in fraction):
        return int(whole)
    return None


def tokenize(source: str, filename: str) -> tuple[Token, ...]:
    """Lex the complete input without truncation, source execution or I/O."""
    if (
        not isinstance(source, str)
        or not isinstance(filename, str)
        or not filename.strip()
    ):
        raise TypeError("source and a nonempty filename must be strings")
    if len(source.encode("utf-8")) > MAX_BYTES:
        reject(
            Token("", SourceLocation(filename, 1, 1)),
            "budget.source",
            f"source exceeds {MAX_BYTES} UTF-8 bytes",
        )
    tokens: list[Token] = []
    offset, line, column = 0, 1, 1
    while offset < len(source):
        location = SourceLocation(filename, line, column)
        if source.startswith("/*", offset) and source.find("*/", offset + 2) < 0:
            reject(Token("/*", location), "token", "unterminated block comment")
        match = _TOKEN.match(source, offset)
        if match is None:
            reject(
                Token(source[offset], location),
                "token",
                f"unsupported character {source[offset]!r}",
            )
        text = match.group()
        if match.lastgroup not in {"space", "comment"}:
            if match.lastgroup == "number" and len(text) > 78:
                reject(
                    Token(text[:16], location),
                    "budget.number",
                    "integer literal exceeds the 256-bit profile's digit budget",
                )
            if match.lastgroup == "scientific":
                mantissa, _, exponent = text.lower().partition("e")
                if len(text) > 78 or abs(int(exponent)) > 1024:
                    reject(
                        Token(text[:16], location),
                        "budget.number",
                        "scientific literal exceeds the numeric budget",
                    )
                real_value = float(text)
                if not math.isfinite(real_value) or (
                    real_value == 0 and any(digit in "123456789" for digit in mantissa)
                ):
                    reject(
                        Token(text, location),
                        "number",
                        "scientific literal must fit a finite real",
                    )
            if len(tokens) >= MAX_TOKENS:
                reject(
                    Token(text, location),
                    "budget.tokens",
                    f"source exceeds {MAX_TOKENS} tokens",
                )
            tokens.append(Token(text, location))
        breaks = text.count("\n")
        column = len(text.rsplit("\n", 1)[-1]) + 1 if breaks else column + len(text)
        line += breaks
        offset = match.end()
    tokens.append(Token("<eof>", SourceLocation(filename, line, column)))
    return tuple(tokens)


@dataclass(frozen=True)
class Expression:
    token: Token
    children: tuple[Expression, ...] = ()
    depth: int = 1


@dataclass(frozen=True)
class Parameter:
    name: Token
    kind: str
    width: int | None
    role: Literal[
        "output",
        "const",
        "local",
        "real",
        "integer",
        "inout",
        "callable",
        "operator",
        "input",
    ] = "output"
    record: str | None = None
    length: int | None = None
    width_expression: Expression | None = None
    fractional_bits: int = 0


@dataclass(frozen=True)
class Record:
    name: Token
    fields: tuple[Parameter, ...]

    @property
    def width(self) -> int:
        return sum((field.width or 0) * (field.length or 1) for field in self.fields)

    def leaves(
        self, layouts: Mapping[str, Record], prefix: str = ""
    ) -> Iterator[Parameter]:
        for field in self.fields:
            if field.record is None:
                yield replace(
                    field, name=replace(field.name, text=prefix + field.name.text)
                )
            else:
                yield from layouts[field.record].leaves(
                    layouts, prefix + field.name.text + "."
                )


@dataclass(frozen=True)
class Statement:
    token: Token
    arguments: tuple[Expression, ...]
    declaration: Parameter | None = None
    within: tuple[Statement, ...] | None = None
    apply: tuple[Statement, ...] = ()
    control_body: tuple[Statement, ...] | None = None
    control_else: tuple[Statement, ...] | None = None
    repeat_index: Token | None = None
    repeat_body: tuple[Statement, ...] = ()
    power_callback: tuple[Token, tuple[Statement, ...]] | None = None
    qpe_body: tuple[Statement, ...] | None = None
    inverse_body: tuple[Statement, ...] | None = None
    static_if_body: tuple[Statement, ...] | None = None
    map_callback: tuple[Token, tuple[Statement, ...]] | None = None
    oracle_callback: tuple[Token, tuple[Statement, ...]] | None = None
    predicate_callback: tuple[Token, Token, tuple[Statement, ...]] | None = None
    amplification: Amplification | None = None
    callback_index: Expression | None = None
    skip_control: tuple[Statement, ...] | None = None
    matrix_row: tuple[Token, ...] | None = None


@dataclass(frozen=True)
class Function:
    name: Token
    parameters: tuple[Parameter, ...]
    statements: tuple[Statement, ...]
    permutation: bool = False
    disable_permutation_check: Token | None = None
    disable_const_checks: Token | None = None
