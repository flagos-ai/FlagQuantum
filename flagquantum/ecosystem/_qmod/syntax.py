"""Adapter-local bounded syntax, discarded at the Compiler model boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NoReturn

from ...compiler._hybrid.model import SourceLocation
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError

MAX_BYTES = 262144
MAX_TOKENS = 4096
MAX_DEPTH = 32
_TOKEN = re.compile(
    r"(?P<space>\s+)|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)"
    r"|(?P<number>[0-9]+)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<symbol>\*\*|\^=|\+=|->|==|[{}()<>,:;=+*\[\].^/\-])"
)
_RESERVED = frozenset(
    {"qfunc", "output", "input", "const", "qbit", "qnum", "true", "false"}
)


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


@dataclass(frozen=True)
class Statement:
    token: Token
    arguments: tuple[Expression, ...]


@dataclass(frozen=True)
class Function:
    name: Token
    parameters: tuple[Parameter, ...]
    statements: tuple[Statement, ...]


class Parser:
    """One source entry; syntax records stay within this private adapter."""

    def __init__(self, tokens: tuple[Token, ...]) -> None:
        self.tokens = tokens
        self.position = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.position]

    def take(self, text: str) -> Token:
        token = self.current
        if token.text != text:
            reject(token, "syntax", f"expected {text!r}, found {token.text!r}")
        if text != "<eof>":
            self.position += 1
        return token

    def name(self) -> Token:
        token = self.current
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", token.text)
            or token.text in _RESERVED
        ):
            reject(token, "name", "expected a non-keyword identifier")
        self.position += 1
        return token

    def width(self) -> int:
        token = self.current
        if not token.text.isdecimal() or not 1 <= int(token.text) <= 256:
            reject(token, "width", "width must be a literal integer in 1..256")
        self.position += 1
        return int(token.text)

    def expression(self, minimum: int = 0, nesting: int = 0) -> Expression:
        token = self.current
        if nesting >= MAX_DEPTH:
            reject(token, "budget.depth", f"expression nesting exceeds {MAX_DEPTH}")
        if token.text == "(":
            self.take("(")
            left = self.expression(nesting=nesting + 1)
            self.take(")")
        elif token.text.isdecimal():
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
        precedence = {"+": 10, "*": 20, "**": 30}
        while precedence.get(self.current.text, -1) >= minimum:
            operator = self.current
            self.position += 1
            level = precedence[operator.text]
            right = self.expression(
                level if operator.text == "**" else level + 1, nesting + 1
            )
            if operator.text == "**" and (right.children or right.token.text != "2"):
                reject(
                    operator,
                    "operator",
                    "only a literal square exponent **2 is admitted",
                )
            depth = max(left.depth, right.depth) + 1
            if depth > MAX_DEPTH:
                reject(
                    operator,
                    "budget.depth",
                    f"expression tree exceeds depth {MAX_DEPTH}",
                )
            left = Expression(operator, (left, right), depth)
        if self.current.text in {"-", "/", "^", "==", "[", ".", "("}:
            reject(
                self.current,
                "operator",
                "this operator, path or numeric form is not admitted",
            )
        return left

    def parse(self) -> Function:
        self.take("qfunc")
        name = self.name()
        if name.text != "main":
            reject(name, "entry", "the initial native profile requires one qfunc main")
        self.take("(")
        parameters: list[Parameter] = []
        while self.current.text != ")":
            if self.current.text != "output":
                reject(
                    self.current,
                    "role",
                    "only output parameters are admitted; Input/Const and borrowed roles are distinct",
                )
            self.take("output")
            parameter = self.name()
            self.take(":")
            kind = self.current
            if kind.text not in {"qbit", "qnum"}:
                reject(
                    kind,
                    "type",
                    "only scalar qbit and unsigned integer qnum are admitted",
                )
            self.position += 1
            width: int | None = 1 if kind.text == "qbit" else None
            if self.current.text == "<":
                if kind.text != "qnum":
                    reject(self.current, "type", "qbit has no numeric modifiers")
                self.take("<")
                width = self.width()
                if self.current.text != ">":
                    reject(
                        self.current,
                        "type",
                        "signed and fractional modifiers are not admitted",
                    )
                self.take(">")
            if self.current.text == "[":
                reject(
                    self.current,
                    "type",
                    "arrays, slices and bind require separate view semantics",
                )
            parameters.append(Parameter(parameter, kind.text, width))
            if self.current.text != ",":
                break
            self.take(",")
            if self.current.text == ")":
                reject(
                    self.current, "syntax", "trailing parameter commas are not admitted"
                )
        self.take(")")
        self.take("{")
        statements: list[Statement] = []
        while self.current.text != "}":
            target = self.name()
            arguments: tuple[Expression, ...]
            if self.current.text == "=":
                operator = self.take("=")
                arguments = (Expression(target), self.expression())
            elif self.current.text == "(":
                operator = target
                self.take("(")
                args: list[Expression] = []
                if self.current.text != ")":
                    args.append(self.expression())
                    while self.current.text == ",":
                        self.take(",")
                        args.append(self.expression())
                self.take(")")
                arguments = tuple(args)
            else:
                reject(
                    self.current,
                    "statement",
                    "only fresh assignment and admitted calls are supported; no locals or in-place assignment",
                )
            self.take(";")
            statements.append(Statement(operator, arguments))
        self.take("}")
        self.take("<eof>")
        return Function(name, tuple(parameters), tuple(statements))
