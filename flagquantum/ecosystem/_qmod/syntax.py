"""Adapter-local bounded syntax, discarded at the Compiler model boundary."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, NoReturn

from ...compiler._hybrid.model import SourceLocation
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from .angles import SCIENTIFIC_LITERAL, parse_literal

if TYPE_CHECKING:
    from .amplification import Amplification

MAX_BYTES = 524288
MAX_TOKENS = 65536
MAX_DEPTH = 40
MAX_EXPRESSION_DEPTH = 128
MAX_SCOPE_DEPTH = 32
MAX_FUNCTIONS = 32
ANGLE_CALLS = frozenset({"PHASE", "RX", "RY", "RZ", "RZZ", "CRX", "CPHASE", "R", "U"})
_TOKEN = re.compile(
    r"(?P<space>\s+)|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)"
    rf"|(?P<scientific>{SCIENTIFIC_LITERAL.pattern})(?![A-Za-z_0-9])"
    r"|(?P<number>[0-9]+)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<symbol>\*\*|\^=|\+=|->|==|!=|<=|>=|[{}()<>,:;=+*%\[\].^/\-&|~])"
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
        "output", "const", "local", "real", "integer", "inout", "callable"
    ] = "output"
    record: str | None = None
    length: int | None = None
    width_expression: Expression | None = None


@dataclass(frozen=True)
class Record:
    name: Token
    fields: tuple[Parameter, ...]

    @property
    def width(self) -> int:
        return sum((field.width or 0) * (field.length or 1) for field in self.fields)


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


@dataclass(frozen=True)
class Function:
    name: Token
    parameters: tuple[Parameter, ...]
    statements: tuple[Statement, ...]
    permutation: bool = False


class Parser:
    """Bounded source definitions; records stay within this private adapter."""

    def __init__(self, tokens: tuple[Token, ...]) -> None:
        self.tokens = tokens
        self.position = 0
        self.callback_depth = 0
        self.records: dict[str, Record] = {}
        self.integer_constants: set[str] = set()
        self.integer_values: dict[str, int] = {}
        self.matrix_constants: set[str] = set()

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
        if token.text == "floor":
            derived = int(self.binary_log_width().token.text)
            if not 1 <= derived <= 256:
                reject(token, "width", "derived width must be in 1..256")
            return derived
        value = (
            int(token.text)
            if token.text.isdecimal()
            else self.integer_values.get(token.text)
        )
        if value is None or not 1 <= value <= 256:
            reject(
                token,
                "width",
                "width must be a literal integer in 1..256 or a bound integer constant",
            )
        self.position += 1
        if token.text.isdecimal() and self.current.text == ".":
            self.take(".")
            fraction = self.current
            if not fraction.text.isdecimal() or fraction.text.strip("0"):
                reject(fraction, "width", "decimal width must be an exact integer")
            self.take(fraction.text)
        return value

    def expression(
        self,
        minimum: int = 0,
        nesting: int = 0,
        *,
        static: bool = False,
        decimal: bool = False,
        difference: bool = False,
        allow_slice: bool = False,
        real: bool = False,
    ) -> Expression:
        token = self.current
        if nesting >= MAX_EXPRESSION_DEPTH:
            reject(
                token,
                "budget.depth",
                f"expression nesting exceeds {MAX_EXPRESSION_DEPTH}",
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
            )
            if operand.depth >= MAX_EXPRESSION_DEPTH:
                reject(
                    token,
                    "budget.depth",
                    f"expression tree exceeds depth {MAX_EXPRESSION_DEPTH}",
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
            )
            self.take(")")
        elif static and token.text == "ceiling":
            left = self.binary_log_width(symbolic=True, nesting=nesting)
        elif (
            static
            and token.text == "floor"
            and self.tokens[self.position + 1].text == "("
        ):
            self.take("floor")
            self.take("(")
            argument = self.expression(nesting=nesting + 1, static=True)
            self.take(")")
            left = Expression(token, (argument,), argument.depth + 1)
        elif (
            real and token.text == "acos" and self.tokens[self.position + 1].text == "("
        ):
            self.take("acos")
            self.take("(")
            argument = self.expression(
                nesting=nesting + 1, decimal=True, difference=True, real=True
            )
            self.take(")")
            left = Expression(token, (argument,), argument.depth + 1)
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
                nesting=nesting + 1, decimal=True, difference=difference
            )
            self.take(",")
            second = self.expression(
                nesting=nesting + 1, decimal=True, difference=difference
            )
            self.take(")")
            depth = max(first.depth, second.depth) + 1
            if depth > MAX_EXPRESSION_DEPTH:
                reject(
                    token,
                    "budget.depth",
                    f"expression tree exceeds depth {MAX_EXPRESSION_DEPTH}",
                )
            left = Expression(token, (first, second), depth)
        while True:
            if not left.children and left.token.text.isdecimal():
                break
            if (
                static
                and self.current.text == ":"
                and [
                    item.text
                    for item in self.tokens[self.position + 1 : self.position + 3]
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
                or (real and self.tokens[self.position + 1].text in {"len", "size"})
            ):
                dot = self.take(".")
                member = Expression(self.name())
                left = Expression(dot, (left, member), left.depth + 1)
            elif self.current.text == "[" and (
                (not left.children)
                or (not static and left.token.text == ":")
                or (
                    self.records
                    and (
                        left.token.text == "."
                        or (
                            left.token.text == "["
                            and left.children[0].token.text == "."
                        )
                    )
                )
            ):
                bracket = self.take("[")
                index = self.expression(nesting=nesting + 1, static=True)
                if allow_slice and self.current.text == ":":
                    colon = self.take(":")
                    stop = self.expression(nesting=nesting + 1, static=True)
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
            if left.depth > MAX_EXPRESSION_DEPTH:
                reject(
                    token,
                    "budget.depth",
                    f"expression tree exceeds depth {MAX_EXPRESSION_DEPTH}",
                )
        precedence = {
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
            if depth > MAX_EXPRESSION_DEPTH:
                reject(
                    operator,
                    "budget.depth",
                    f"expression tree exceeds depth {MAX_EXPRESSION_DEPTH}",
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

    def binary_log_width(
        self, *, symbolic: bool = False, nesting: int = 0
    ) -> Expression:
        token = self.take(self.current.text)
        self.take("(")
        self.take("log")
        self.take("(")
        if symbolic:
            operand = self.expression(nesting=nesting + 1, static=True)
            if operand.depth >= MAX_EXPRESSION_DEPTH:
                reject(
                    token, "budget.depth", "binary log width exceeds expression depth"
                )
            for part in (",", "2", ")", ")"):
                self.take(part)
            return Expression(
                Token("$ceil_log2", token.location), (operand,), operand.depth + 1
            )
        value = self.current
        size = (
            int(value.text)
            if value.text.isdecimal()
            else self.integer_values.get(value.text)
        )
        if size is None or size < 2:
            reject(
                value,
                "width",
                "binary log width requires an integer literal of at least two",
            )
        self.take(value.text)
        self.take(",")
        self.take("2")
        self.take(")")
        self.take(")")
        width = (
            size.bit_length() - 1 if token.text == "floor" else (size - 1).bit_length()
        )
        return Expression(Token(str(width), token.location))

    def angle_argument(self) -> Expression:
        return parse_literal(self)

    def real_angle(self) -> Expression:
        return self.expression(
            decimal=True, difference=True, real=True, allow_slice=True
        )

    def signed_angle_argument(self) -> Expression:
        if self.current.text == "-":
            return Expression(self.take("-"), (self.angle_argument(),))
        return self.angle_argument()

    def preparation_literal(self, *, allow_fraction: bool) -> Expression:
        grouped = self.current.text == "("
        if grouped:
            self.take("(")
        numerator = self.current
        if (
            allow_fraction
            and numerator.text.isdecimal()
            and self.tokens[self.position + 1].text == "/"
        ):
            self.position += 1
            slash = self.take("/")
            denominator = self.current
            if not denominator.text.isdecimal():
                reject(
                    denominator,
                    "probabilities",
                    "fraction requires an integer denominator",
                )
            self.position += 1
            value = Expression(slash, (Expression(numerator), Expression(denominator)))
        else:
            value = self.signed_angle_argument()
        if grouped:
            self.take(")")
        return value

    def pauli_axis(self) -> Expression:
        self.take("Pauli")
        self.take(":")
        self.take(":")
        axis = self.name()
        if axis.text not in {"I", "X", "Y", "Z"}:
            reject(axis, "pauli", "expected Pauli::I, X, Y or Z")
        return Expression(axis)

    def matrix_entry(self, depth: int = 0) -> Expression:
        entry = self.matrix_atom(depth)
        if self.current.text in {"+", "-"}:
            operator = self.take(self.current.text)
            return Expression(operator, (entry, self.matrix_atom(depth)))
        return entry

    def matrix_atom(self, depth: int) -> Expression:
        token = self.current
        if depth >= MAX_DEPTH:
            reject(token, "budget.depth", "matrix entry exceeds depth budget")
        if token.text in {"+", "-"}:
            return Expression(self.take(token.text), (self.matrix_atom(depth + 1),))
        if token.text == "(":
            self.take("(")
            entry = self.matrix_entry(depth + 1)
            self.take(")")
            return entry
        if not token.text.isdecimal():
            reject(token, "matrix", "expected a numeric matrix literal")
        if self.tokens[self.position + 1].text == ".":
            number = self.angle_argument()
        else:
            self.position += 1
            number = Expression(token)
        suffix = self.current
        if (
            suffix.text == "j"
            and suffix.location.line == token.location.line
            and suffix.location.column == token.location.column + len(number.token.text)
        ):
            return Expression(self.take("j"), (number,))
        return number

    def linear_rotation_basis(self) -> Expression:
        if self.current.text == "Pauli":
            return self.pauli_axis()
        token = self.current
        if not token.text.isdecimal():
            reject(token, "pauli", "expected a Pauli axis or integer code")
        self.position += 1
        return Expression(token)

    def literal_array(self, item: Callable[[], Expression]) -> Expression:
        start = self.take("[")
        items = []
        while self.current.text != "]":
            items.append(item())
            if self.current.text != ",":
                break
            self.take(",")
        self.take("]")
        return Expression(start, tuple(items))

    def pauli_record(
        self, kind: str, fields: dict[str, Callable[[], Expression]]
    ) -> Expression:
        record = self.take(kind)
        self.take("{")
        values = {}
        for position in range(len(fields)):
            if position:
                self.take(",")
            field = self.name()
            if field.text in values or field.text not in fields:
                reject(
                    field,
                    "pauli",
                    f"{kind} requires distinct fields: {', '.join(fields)}",
                )
            self.take("=")
            values[field.text] = fields[field.text]()
        self.take("}")
        return Expression(record, tuple(values[name] for name in fields))

    def pauli_terms(self) -> Expression:
        return self.literal_array(
            lambda: self.pauli_record(
                "PauliTerm",
                {
                    "pauli": lambda: self.literal_array(self.pauli_axis),
                    "coefficient": self.signed_angle_argument,
                },
            )
        )

    def indexed_pauli(self) -> Expression:
        return self.pauli_record(
            "IndexedPauli", {"pauli": self.pauli_axis, "index": self.expression}
        )

    def sparse_pauli_coefficient(self) -> Expression:
        grouped = self.current.text == "("
        if grouped:
            self.take("(")
        sign = self.take("-") if self.current.text == "-" else None
        token = self.current
        if token.text.isdecimal() and self.tokens[self.position + 1].text != ".":
            self.position += 1
            real = Expression(token)
        else:
            real = self.angle_argument()
        if grouped:
            self.take(")")
        if self.current.text == "+":
            suffix = self.take("+")
            zero = self.take("0")
            unit = self.take("j")
            if (
                not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", real.token.text)
                or real.children
                or unit.location.line != zero.location.line
                or unit.location.column != zero.location.column + 1
            ):
                reject(
                    suffix,
                    "pauli",
                    "Pauli coefficients admit only a literal + 0j suffix",
                )
        return real if sign is None else Expression(sign, (real,))

    def sparse_pauli_term(self) -> Expression:
        return self.pauli_record(
            "SparsePauliTerm",
            {
                "paulis": lambda: self.literal_array(self.indexed_pauli),
                "coefficient": self.sparse_pauli_coefficient,
            },
        )

    def sparse_pauli_op(self) -> Expression:
        return self.pauli_record(
            "SparsePauliOp",
            {
                "terms": lambda: self.literal_array(self.sparse_pauli_term),
                "num_qubits": self.expression,
            },
        )

    def scalar_type(
        self,
        *,
        allow_qubit_array: bool = False,
        allow_phase: bool = False,
        allow_fixed_array: bool = False,
        allow_numeric_array: bool = False,
        allow_signed_integer: bool = False,
    ) -> tuple[str, int | None]:
        kind = self.current
        if kind.text not in {"qbit", "qnum"}:
            reject(
                kind, "type", "only scalar qbit and unsigned integer qnum are admitted"
            )
        self.position += 1
        width: int | None = 1 if kind.text == "qbit" else None
        if self.current.text == "<":
            if kind.text != "qnum":
                reject(self.current, "type", "qbit has no numeric modifiers")
            self.take("<")
            width = self.width()
            if self.current.text == ",":
                self.take(",")
                from .qpe import parse_fraction

                phase_kind = parse_fraction(
                    self, width, allow_phase, allow_signed_integer=allow_signed_integer
                )
                if phase_kind is not None:
                    return phase_kind, width
            if self.current.text != ">":
                reject(
                    self.current,
                    "type",
                    "signed and fractional modifiers are not admitted",
                )
            self.take(">")
        if (
            allow_qubit_array
            and kind.text == "qbit"
            and self.current.text == "["
            and self.tokens[self.position + 1].text == "]"
        ):
            self.take("[")
            self.take("]")
            return "qbit[]", None
        if kind.text == "qbit" and self.current.text == "[" and allow_fixed_array:
            self.take("[")
            width = self.width()
            self.take("]")
            return "qbit[]", width
        if self.current.text == "[" and kind.text == "qnum" and allow_numeric_array:
            return kind.text, width
        if self.current.text == "[":
            reject(
                self.current,
                "type",
                "arrays, slices and bind require separate view semantics",
            )
        return kind.text, width

    def function(self) -> Function:
        permutation = self.current.text == "qperm"
        self.take("qperm" if permutation else "qfunc")
        name = self.name()
        self.take("(")
        parameters: list[Parameter] = []
        while self.current.text != ")":
            role: Literal["output", "const", "real", "integer", "inout", "callable"]
            if self.current.text == "output":
                self.take("output")
                role = "output"
                if self.current.text == "const":
                    reject(
                        self.current,
                        "role",
                        "output const is outside the admitted roles",
                    )
            elif self.current.text == "const" and name.text != "main":
                self.take("const")
                role = "const"
            elif [
                token.text
                for token in self.tokens[self.position + 1 : self.position + 3]
            ] in ([":", "real"], [":", "int"]):
                role = (
                    "integer"
                    if self.tokens[self.position + 2].text == "int"
                    else "real"
                )
            elif (
                name.text != "main"
                and self.position + 2 < len(self.tokens)
                and self.tokens[self.position + 2].text in self.records
            ):
                role = "inout"
            elif name.text != "main" and [
                token.text
                for token in self.tokens[self.position + 1 : self.position + 3]
            ] == [":", "qfunc"]:
                role = "callable"
            elif name.text != "main" and [
                token.text
                for token in self.tokens[self.position + 1 : self.position + 3]
            ] in ([":", "qbit"], [":", "qnum"]):
                role = "inout"
            else:
                reject(
                    self.current,
                    "role",
                    "expected an Output, a classical real, or a supported helper port",
                )
            parameter = self.name()
            self.take(":")
            record = None
            if self.current.text in self.records:
                if role not in {"output", "inout", "const"}:
                    reject(
                        self.current,
                        "role",
                        "records require Output or writable helper ports",
                    )
                record = self.take(self.current.text).text
                kind, width = "qbit[]", self.records[record].width
            elif role == "callable":
                from .quantum_callbacks import parse_signature

                kind, width = parse_signature(self), None
            elif role in {"real", "integer"}:
                kind = "int" if role == "integer" else "real"
                self.take(kind)
                width = None
                if self.current.text == "[":
                    self.take("[")
                    if name.text == "main" or self.current.text != "]":
                        width = self.width()
                    self.take("]")
                    kind += "[]"
            else:
                kind, width = self.scalar_type(
                    allow_qubit_array=name.text == "main" or role in {"inout", "const"},
                    allow_phase=True,
                    allow_signed_integer=name.text == "main" and role == "output",
                    allow_fixed_array=role in {"inout", "output", "const"}
                    or name.text == "main",
                )
                if kind == "qphase_signed" and name.text != "main":
                    reject(
                        parameter,
                        "type",
                        "signed fixed-point helper ports are not admitted",
                    )
            parameters.append(Parameter(parameter, kind, width, role, record))
            if self.current.text != ",":
                break
            self.take(",")
            if self.current.text == ")":
                reject(
                    self.current, "syntax", "trailing parameter commas are not admitted"
                )
        self.take(")")
        return Function(name, tuple(parameters), self.block(), permutation=permutation)

    def call_argument(self) -> Expression:
        if self.current.text == "lambda":
            from .quantum_callbacks import parse_lambda

            return parse_lambda(self)
        if self.current.text == "[":
            if self.tokens[self.position + 1].text == "lambda":
                from .quantum_callbacks import parse_lambda

                array = self.literal_array(partial(parse_lambda, self))
            else:
                array = self.literal_array(
                    lambda: self.expression(
                        decimal=True, difference=True, real=True, allow_slice=True
                    )
                )
            return Expression(Token("[]", array.token.location), array.children)
        if self.current.text == "{":
            opening = self.take("{")
            items = [self.expression()]
            while self.current.text == ",":
                self.take(",")
                items.append(self.expression())
            self.take("}")
            return Expression(opening, tuple(items))
        return self.expression(
            decimal=True, difference=True, real=True, allow_slice=True
        )

    def binding_group(self) -> Expression:
        if self.current.text != "{":
            return Expression(self.name())
        token = self.take("{")
        elements = [Expression(self.name())]
        while self.current.text == ",":
            self.take(",")
            elements.append(Expression(self.name()))
        self.take("}")
        return Expression(token, tuple(elements))

    def block(
        self,
        *,
        allow_scopes: bool = True,
        scope_depth: int = 0,
        power_body: bool = False,
        static_loop_body: bool = False,
        control_actions: bool = False,
        binding_preparation: bool = False,
        binding_actions: bool = False,
    ) -> tuple[Statement, ...]:
        if scope_depth > MAX_SCOPE_DEPTH:
            reject(
                self.current,
                "budget.depth",
                f"scope nesting exceeds depth {MAX_SCOPE_DEPTH}",
            )
        self.take("{")
        statements: list[Statement] = []
        while self.current.text != "}":
            if self.current.text == "{" or self.tokens[self.position + 1].text == "->":
                if scope_depth and not binding_preparation:
                    reject(self.current, "scope", "bindings require function scope")
                source = self.binding_group()
                operator = self.take("->")
                destination = self.binding_group()
                self.take(";")
                statements.append(Statement(operator, (source, destination)))
                continue
            if (
                self.current.text == "invert"
                and self.tokens[self.position + 1].text == "{"
            ):
                token = self.take("invert")
                if scope_depth:
                    if scope_depth >= MAX_SCOPE_DEPTH:
                        reject(
                            token,
                            "budget.depth",
                            f"scope nesting exceeds depth {MAX_SCOPE_DEPTH}",
                        )
                body = self.block(allow_scopes=False, scope_depth=scope_depth + 1)
                statements.append(Statement(token, (), inverse_body=body))
                continue
            if (
                self.current.text == "within"
                and self.tokens[self.position + 1].text == "{"
            ):
                from .bindings import parse_within

                statements.append(parse_within(self, allow_scopes, scope_depth))
                continue
            if (
                self.current.text in {"repeat", "power"}
                and self.tokens[self.position + 1].text == "("
            ):
                token = self.take(self.current.text)
                if scope_depth and not (
                    static_loop_body or binding_actions or control_actions
                ):
                    reject(
                        token,
                        "scope",
                        "static loops are admitted only at function scope",
                    )
                self.take("(")
                if token.text == "repeat":
                    index = self.name()
                    self.take(":")
                else:
                    index = Token(
                        f"$power{token.location.line}_{token.location.column}",
                        token.location,
                    )
                count = self.expression(static=True)
                self.take(")")
                body = self.block(
                    allow_scopes=False,
                    scope_depth=scope_depth + 1,
                    static_loop_body=True,
                    binding_actions=binding_actions,
                )
                statements.append(
                    Statement(token, (count,), repeat_index=index, repeat_body=body)
                )
                continue
            if self.current.text == "if" and self.tokens[self.position + 1].text == "(":
                token = self.take("if")
                if scope_depth and not static_loop_body:
                    reject(
                        token,
                        "scope",
                        "static if requires function scope or a static loop",
                    )
                self.take("(")
                if self.current.text in {"True", "False", "true", "false"}:
                    literal = self.take(self.current.text)
                    self.take(")")
                    first = self.block(
                        allow_scopes=allow_scopes,
                        scope_depth=scope_depth + 1,
                        static_loop_body=static_loop_body,
                    )
                    second: tuple[Statement, ...] = ()
                    if self.current.text == "else":
                        self.take("else")
                        second = self.block(
                            allow_scopes=allow_scopes,
                            scope_depth=scope_depth + 1,
                            static_loop_body=static_loop_body,
                        )
                    selected = first if literal.text in {"True", "true"} else second
                    if any(item.declaration is not None for item in (*first, *second)):
                        reject(
                            token,
                            "scope",
                            "static Boolean branches cannot declare local storage",
                        )
                    statements.extend(selected)
                    continue
                left = self.expression(static=True)
                comparison = self.current
                if comparison.text not in _COMPARISONS:
                    reject(
                        comparison,
                        "expression",
                        "static if requires an integer comparison",
                    )
                self.take(comparison.text)
                right = self.expression(static=True)
                self.take(")")
                body = self.block(allow_scopes=False, scope_depth=1)
                condition = Expression(comparison, (left, right))
                statements.append(Statement(token, (condition,), static_if_body=body))
                continue
            target = self.name()
            arguments: tuple[Expression, ...]
            if target.text == "free" and scope_depth:
                reject(target, "scope", "free requires function scope")
            if (
                target.text in {"grover_operator", "grover_search", "phase_oracle"}
                and self.current.text == "("
            ):
                from .oracles import parse_oracle_call

                statements.append(parse_oracle_call(self, target, scope_depth))
                continue
            if (
                target.text == "apply_to_all"
                and self.tokens[self.position + 1].text == "lambda"
            ):
                self.take("(")
                self.take("lambda")
                self.take("(")
                parameter = self.name()
                self.take(")")
                body = self.block(allow_scopes=False, scope_depth=scope_depth + 1)
                self.take(",")
                target_array = self.call_argument()
                self.take(")")
                self.take(";")
                statements.append(
                    Statement(target, (target_array,), map_callback=(parameter, body))
                )
                continue
            if target.text in {"qpe", "qpe_flexible"} and self.current.text == "(":
                from .qpe import parse_qpe_call

                statements.append(parse_qpe_call(self, target, scope_depth))
                continue
            if self.current.text == ":":
                token = self.take(":")
                if scope_depth:
                    reject(
                        token, "scope", "declare locals at function level before within"
                    )
                record = None
                local_width: int | None
                width_expression = None
                if (
                    self.current.text == "qnum"
                    and self.tokens[self.position + 1].text == "<"
                    and self.position + 3 < len(self.tokens)
                    and self.tokens[self.position + 3].text == "."
                    and not self.tokens[self.position + 2].text.isdecimal()
                ):
                    self.take("qnum")
                    self.take("<")
                    width_expression = self.expression(static=True)
                    if self.current.text == ",":
                        self.take(",")
                        sign = self.current
                        if sign.text not in {"UNSIGNED", "False"}:
                            reject(
                                sign, "type", "derived width requires unsigned storage"
                            )
                        self.take(sign.text)
                        self.take(",")
                        self.take("0")
                    self.take(">")
                    kind, local_width = "qnum", None
                elif self.current.text in self.records:
                    record = self.name().text
                    kind, local_width = "qbit[]", self.records[record].width
                else:
                    kind, local_width = self.scalar_type(
                        allow_fixed_array=True, allow_qubit_array=True, allow_phase=True
                    )
                    if kind == "qphase_signed":
                        reject(
                            target, "type", "signed fixed-point locals are not admitted"
                        )
                self.take(";")
                statements.append(
                    Statement(
                        token,
                        (),
                        declaration=Parameter(
                            target,
                            kind,
                            local_width,
                            "local",
                            record,
                            width_expression=width_expression,
                        ),
                    )
                )
                continue
            if self.current.text == "[":
                self.take("[")
                callback_index = self.expression(static=True)
                for part in ("]", "(", ")", ";"):
                    self.take(part)
                statements.append(Statement(target, (), callback_index=callback_index))
                continue
            if self.current.text in {"=", "^=", "+="}:
                operator = self.take(self.current.text)
                rhs = (
                    self.signed_angle_argument()
                    if operator.text == "^=" and self.current.text == "-"
                    else self.expression(
                        decimal=operator.text == "=",
                        difference=True,
                    )
                )
                arguments = (Expression(target), rhs)
            elif self.current.text == "(":
                operator = target
                self.take("(")
                args: list[Expression] = []
                if self.current.text != ")":
                    if target.text == "multi_suzuki_trotter":
                        args.append(self.literal_array(self.sparse_pauli_op))
                    elif target.text in {"exponentiate", "qdrift", "suzuki_trotter"}:
                        if self.current.text == "SparsePauliOp":
                            args.append(self.sparse_pauli_op())
                        elif (
                            target.text == "suzuki_trotter" and self.current.text == "["
                        ):
                            args.append(self.pauli_terms())
                        else:
                            args.append(Expression(self.name()))
                    elif target.text == "allocate" and self.current.text == "ceiling":
                        args.append(self.expression(static=True))
                    elif (
                        target.text == "grover_diffuser"
                        and self.current.text == "lambda"
                    ):
                        from .quantum_callbacks import parse_lambda

                        args.append(parse_lambda(self, preparation=True))
                    elif target.text == "prepare_exponential_state":
                        args.append(self.signed_angle_argument())
                    elif target.text == "unitary":
                        args.append(
                            self.literal_array(
                                lambda: self.literal_array(self.matrix_entry)
                            )
                            if self.current.text == "["
                            else Expression(self.name())
                        )
                    elif target.text == "linear_pauli_rotations":
                        args.append(self.literal_array(self.linear_rotation_basis))
                    elif target.text == "select_rotation":
                        args.append(self.pauli_axis())
                    elif target.text == "single_pauli_exponent":
                        args.append(self.literal_array(self.pauli_axis))
                    elif target.text in PREPARATIONS:
                        args.append(
                            self.literal_array(
                                partial(
                                    self.preparation_literal,
                                    allow_fraction=target.text
                                    in {"prepare_state", "inplace_prepare_state"},
                                )
                            )
                            if self.current.text == "["
                            else Expression(self.name())
                        )
                    elif target.text == "control":
                        args.append(self.expression(difference=True, allow_slice=True))
                    elif target.text == "phase":
                        args.append(
                            self.expression(decimal=True, difference=True, real=True)
                        )
                    elif target.text in ANGLE_CALLS:
                        args.append(self.real_angle())
                    else:
                        args.append(self.call_argument())
                    while self.current.text == ",":
                        self.take(",")
                        if (
                            len(args) == 2
                            and target.text == "allocate"
                            and self.current.text == "ceiling"
                        ):
                            args.append(self.expression(static=True))
                            continue
                        if len(args) == 1 and target.text == "multi_suzuki_trotter":
                            args.append(self.call_argument())
                            continue
                        if target.text == "linear_pauli_rotations" and len(args) in {
                            1,
                            2,
                        }:
                            args.append(self.literal_array(self.signed_angle_argument))
                            continue
                        if len(args) == 1 and target.text in {
                            "exponentiate",
                            "qdrift",
                            "prepare_amplitudes",
                            "prepare_state",
                            "suzuki_trotter",
                            "single_pauli_exponent",
                        }:
                            args.append(
                                self.signed_angle_argument()
                                if target.text in PREPARATIONS
                                else self.real_angle()
                            )
                            continue
                        args.append(
                            self.real_angle()
                            if (target.text == "U" and len(args) < 4)
                            or (len(args) == 1 and target.text in {"R", "phase"})
                            else self.call_argument()
                        )
                self.take(")")
                arguments = tuple(args)
            else:
                reject(
                    self.current,
                    "statement",
                    "only fresh assignment, scalar declarations and admitted calls are supported",
                )
            if operator.text == "control" and self.current.text == "{":
                if scope_depth >= MAX_SCOPE_DEPTH:
                    reject(
                        operator,
                        "budget.depth",
                        f"scope nesting exceeds depth {MAX_SCOPE_DEPTH}",
                    )
                body = self.block(
                    allow_scopes=False,
                    scope_depth=scope_depth + 1,
                    control_actions=True,
                )
                alternate = None
                if self.current.text == "else":
                    self.take("else")
                    alternate = self.block(
                        allow_scopes=False,
                        scope_depth=scope_depth + 1,
                        control_actions=True,
                    )
                statements.append(
                    Statement(
                        operator, arguments, control_body=body, control_else=alternate
                    )
                )
            else:
                self.take(";")
                statements.append(Statement(operator, arguments))
        self.take("}")
        return tuple(statements)

    def record(self) -> None:
        self.take("qstruct")
        name = self.name()
        if name.text in self.records:
            reject(name, "name", "duplicate record type")
        self.take("{")
        fields: list[Parameter] = []
        while self.current.text != "}":
            field = self.name()
            if field.text in {"size", "len"} or any(
                item.name.text == field.text for item in fields
            ):
                reject(
                    field,
                    "name",
                    "record field names must be unique and cannot shadow size or len",
                )
            self.take(":")
            if self.current.text not in {"qbit", "qnum"}:
                reject(
                    self.current,
                    "type",
                    "flat record fields require qbit or unsigned qnum",
                )
            kind, width = self.scalar_type(
                allow_fixed_array=True, allow_numeric_array=True
            )
            if width is None:
                reject(field, "width", "record numeric fields require a fixed width")
            length = None
            if kind in {"qnum", "qbit[]"} and self.current.text == "[":
                self.take("[")
                length = self.width()
                self.take("]")
                kind = "qnum[]" if kind == "qnum" else "qbit[][]"
            if self.current.text == "[":
                reject(
                    self.current,
                    "type",
                    "record numeric fields support one array dimension; bit fields support two array dimensions",
                )
            self.take(";")
            fields.append(Parameter(field, kind, width, length=length))
            if sum((item.width or 0) * (item.length or 1) for item in fields) > 256:
                reject(field, "budget.record", "record exceeds 256 bits")
        self.take("}")
        if not fields:
            reject(name, "type", "record requires at least one field")
        self.records[name.text] = Record(name, tuple(fields))

    def constant_integer(self, node: Expression) -> int:
        from .integers import checked_integer, integer_operation

        token = node.token
        if not node.children:
            if token.text.isdecimal():
                return checked_integer(int(token.text), token)
            if token.text in self.integer_values:
                return self.integer_values[token.text]
        if token.text == "-" and len(node.children) == 1:
            return -self.constant_integer(node.children[0])
        if token.text in {"+", "-", "*", "%", "**"}:
            left, right = (self.constant_integer(child) for child in node.children)
            return integer_operation(token, left, right)
        reject(token, "constant", "int constants require a bounded integer expression")

    def parse(
        self,
    ) -> tuple[tuple[Function, ...], tuple[tuple[Token, Expression], ...]]:
        if self.current.text == "<eof>":
            self.take("qfunc")
        functions: list[Function] = []
        constants: list[tuple[Token, Expression]] = []
        while self.current.text != "<eof>":
            if len(functions) + len(constants) + len(self.records) >= MAX_FUNCTIONS:
                reject(
                    self.current, "budget.functions", "source exceeds 32 definitions"
                )
            if self.current.text == "qstruct":
                self.record()
                continue
            kind = (
                self.tokens[self.position + 2].text
                if self.position + 2 < len(self.tokens)
                else ""
            )
            if (
                self.current.text != "qfunc"
                and self.tokens[self.position + 1].text == ":"
                and kind in {"PauliTerm", "SparsePauliOp", "real", "int"}
            ):
                name = self.name()
                self.take(":")
                self.take(kind)
                array = kind == "real" and self.current.text == "["
                if array:
                    self.take("[")
                    self.take("]")
                matrix = array and self.current.text == "["
                if matrix:
                    self.take("[")
                    self.take("]")
                    self.matrix_constants.add(name.text)
                if kind == "PauliTerm":
                    self.take("[")
                    self.take("]")
                self.take("=")
                if matrix:
                    value = self.literal_array(
                        lambda: self.literal_array(self.matrix_entry)
                    )
                elif kind == "SparsePauliOp":
                    value = self.sparse_pauli_op()
                elif kind == "PauliTerm":
                    literal = self.pauli_terms()
                    value = Expression(
                        Token("PauliTerm[]", literal.token.location), literal.children
                    )
                elif array:
                    value = self.literal_array(
                        partial(self.preparation_literal, allow_fraction=False)
                    )
                elif kind == "int":
                    self.integer_constants.add(name.text)
                    value = self.expression(static=True, decimal=True)
                    self.integer_values[name.text] = self.constant_integer(value)
                else:
                    value = self.signed_angle_argument()
                constants.append((name, value))
                self.take(";")
            else:
                functions.append(self.function())
        self.take("<eof>")
        return tuple(functions), tuple(constants)
