"""Adapter-local bounded syntax, discarded at the Compiler model boundary."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NoReturn

from ...compiler._hybrid.model import SourceLocation
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError

MAX_BYTES = 262144
MAX_TOKENS = 4096
MAX_DEPTH = 32
MAX_FUNCTIONS = 32
ANGLE_CALLS = frozenset({"PHASE", "RX", "RY", "RZ", "RZZ", "CRX", "R", "U"})
_TOKEN = re.compile(
    r"(?P<space>\s+)|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)"
    r"|(?P<number>[0-9]+)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<symbol>\*\*|\^=|\+=|->|==|!=|<=|>=|[{}()<>,:;=+*%\[\].^/\-&|~])"
)
_RESERVED = frozenset(
    {
        "qfunc",
        "output",
        "input",
        "const",
        "qbit",
        "qnum",
        "true",
        "false",
    }
)
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
    role: Literal["output", "const", "local", "real", "inout"] = "output"


@dataclass(frozen=True)
class Statement:
    token: Token
    arguments: tuple[Expression, ...]
    declaration: Parameter | None = None
    within: tuple[Statement, ...] | None = None
    apply: tuple[Statement, ...] = ()
    control_body: tuple[Statement, ...] | None = None
    repeat_index: Token | None = None
    repeat_body: tuple[Statement, ...] = ()


@dataclass(frozen=True)
class Function:
    name: Token
    parameters: tuple[Parameter, ...]
    statements: tuple[Statement, ...]


class Parser:
    """Bounded source definitions; records stay within this private adapter."""

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

    def expression(
        self,
        minimum: int = 0,
        nesting: int = 0,
        *,
        static: bool = False,
        decimal: bool = False,
        difference: bool = False,
    ) -> Expression:
        token = self.current
        if nesting >= MAX_DEPTH:
            reject(token, "budget.depth", f"expression nesting exceeds {MAX_DEPTH}")
        if not static and (token.text == "~" or (difference and token.text == "-")):
            self.take(token.text)
            operand = self.expression(
                25, nesting + 1, decimal=decimal, difference=difference
            )
            if operand.depth >= MAX_DEPTH:
                reject(
                    token, "budget.depth", f"expression tree exceeds depth {MAX_DEPTH}"
                )
            left = Expression(token, (operand,), operand.depth + 1)
        elif token.text == "(":
            self.take("(")
            left = self.expression(
                nesting=nesting + 1,
                static=static,
                decimal=decimal,
                difference=difference,
            )
            self.take(")")
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
        if not left.children and not left.token.text.isdecimal():
            if static and self.current.text == ".":
                dot = self.take(".")
                self.take("len")
                left = Expression(dot, (left,), 2)
            elif not static and token.text in EXTREMA and self.current.text == "(":
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
                if depth > MAX_DEPTH:
                    reject(
                        token,
                        "budget.depth",
                        f"expression tree exceeds depth {MAX_DEPTH}",
                    )
                left = Expression(token, (first, second), depth)
            elif not static and self.current.text == "[":
                bracket = self.take("[")
                index = self.expression(nesting=nesting + 1, static=True)
                self.take("]")
                left = Expression(bracket, (left, index), index.depth + 1)
        precedence = {
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
            precedence = {"+": 10, "-": 10}
        elif difference:
            precedence["-"] = 10
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
            )
            if operator.text == "**" and (
                right.children
                or not right.token.text.isdecimal()
                or not 1 <= int(right.token.text) <= 8
            ):
                reject(
                    operator,
                    "operator",
                    "power requires a literal exponent in 1..8",
                )
            depth = max(left.depth, right.depth) + 1
            if depth > MAX_DEPTH:
                reject(
                    operator,
                    "budget.depth",
                    f"expression tree exceeds depth {MAX_DEPTH}",
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

    def angle_argument(self) -> Expression:
        integer = self.current
        if not integer.text.isdecimal() or self.tokens[self.position + 1].text != ".":
            return self.expression()
        self.position += 1
        dot = self.take(".")
        fraction = self.current
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
        self.position += 1
        return Expression(Token(text, integer.location))

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

    def scalar_type(self, *, allow_qubit_array: bool = False) -> tuple[str, int | None]:
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
                if self.current.text != "False":
                    reject(self.current, "type", "signed modifiers are not admitted")
                self.take("False")
                self.take(",")
                if self.current.text != "0":
                    reject(
                        self.current, "type", "fractional modifiers are not admitted"
                    )
                self.take("0")
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
        if self.current.text == "[":
            reject(
                self.current,
                "type",
                "arrays, slices and bind require separate view semantics",
            )
        return kind.text, width

    def function(self) -> Function:
        self.take("qfunc")
        name = self.name()
        self.take("(")
        parameters: list[Parameter] = []
        while self.current.text != ")":
            role: Literal["output", "const", "real", "inout"]
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
            ] == [":", "real"]:
                role = "real"
            elif name.text != "main" and [
                token.text
                for token in self.tokens[self.position + 1 : self.position + 5]
            ] == [":", "qbit", "[", "]"]:
                role = "inout"
            else:
                reject(
                    self.current,
                    "role",
                    "expected an Output, a classical real, or a supported helper port",
                )
            parameter = self.name()
            self.take(":")
            if role == "real":
                self.take("real")
                kind, width = "real", None
                if name.text == "main" and self.current.text == "[":
                    self.take("[")
                    width = self.width()
                    self.take("]")
                    kind = "real[]"
            else:
                kind, width = self.scalar_type(
                    allow_qubit_array=name.text == "main" or role == "inout"
                )
            if role == "const" and width is None:
                reject(
                    parameter, "width", "Const qnum requires an explicit unsigned width"
                )
            parameters.append(Parameter(parameter, kind, width, role))
            if self.current.text != ",":
                break
            self.take(",")
            if self.current.text == ")":
                reject(
                    self.current, "syntax", "trailing parameter commas are not admitted"
                )
        self.take(")")
        return Function(name, tuple(parameters), self.block())

    def block(
        self,
        *,
        allow_scopes: bool = True,
        scope_depth: int = 0,
    ) -> tuple[Statement, ...]:
        self.take("{")
        statements: list[Statement] = []
        while self.current.text != "}":
            if (
                self.current.text == "within"
                and self.tokens[self.position + 1].text == "{"
            ):
                token = self.take("within")
                if not allow_scopes:
                    reject(
                        token, "scope", "nested scopes belong in apply, not preparation"
                    )
                if scope_depth >= MAX_DEPTH:
                    reject(token, "budget.depth", "scope nesting exceeds depth 32")
                preparation = self.block(
                    allow_scopes=False, scope_depth=scope_depth + 1
                )
                self.take("apply")
                actions = self.block(scope_depth=scope_depth + 1)
                statements.append(
                    Statement(token, (), within=preparation, apply=actions)
                )
                continue
            if (
                self.current.text == "repeat"
                and self.tokens[self.position + 1].text == "("
            ):
                token = self.take("repeat")
                if scope_depth:
                    reject(token, "scope", "repeat is admitted only at function scope")
                self.take("(")
                index = self.name()
                self.take(":")
                count = self.expression(static=True)
                self.take(")")
                body = self.block(allow_scopes=False, scope_depth=scope_depth + 1)
                statements.append(
                    Statement(token, (count,), repeat_index=index, repeat_body=body)
                )
                continue
            target = self.name()
            arguments: tuple[Expression, ...]
            if self.current.text == ":":
                token = self.take(":")
                if scope_depth:
                    reject(
                        token, "scope", "declare locals at function level before within"
                    )
                kind, width = self.scalar_type()
                self.take(";")
                statements.append(
                    Statement(
                        token, (), declaration=Parameter(target, kind, width, "local")
                    )
                )
                continue
            if self.current.text in {"=", "^="}:
                operator = self.take(self.current.text)
                rhs = (
                    self.signed_angle_argument()
                    if operator.text == "^=" and self.current.text == "-"
                    else self.expression(
                        decimal=operator.text == "=", difference=operator.text == "="
                    )
                )
                arguments = (Expression(target), rhs)
            elif self.current.text == "(":
                operator = target
                self.take("(")
                args: list[Expression] = []
                if self.current.text != ")":
                    if target.text in {"exponentiate", "qdrift", "suzuki_trotter"}:
                        if self.current.text == "SparsePauliOp":
                            args.append(self.sparse_pauli_op())
                        elif (
                            target.text == "suzuki_trotter" and self.current.text == "["
                        ):
                            args.append(self.pauli_terms())
                        else:
                            args.append(Expression(self.name()))
                    elif target.text == "prepare_exponential_state":
                        args.append(self.signed_angle_argument())
                    elif target.text == "unitary":
                        args.append(
                            self.literal_array(
                                lambda: self.literal_array(self.matrix_entry)
                            )
                        )
                    elif target.text == "linear_pauli_rotations":
                        args.append(self.literal_array(self.linear_rotation_basis))
                    elif target.text == "single_pauli_exponent":
                        args.append(self.literal_array(self.pauli_axis))
                    elif target.text in {"prepare_amplitudes", "prepare_state"}:
                        args.append(
                            self.literal_array(
                                partial(
                                    self.preparation_literal,
                                    allow_fraction=target.text == "prepare_state",
                                )
                            )
                            if self.current.text == "["
                            else Expression(self.name())
                        )
                    elif target.text in ANGLE_CALLS:
                        args.append(self.angle_argument())
                    else:
                        args.append(self.expression())
                    while self.current.text == ",":
                        self.take(",")
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
                            args.append(self.signed_angle_argument())
                            continue
                        args.append(
                            self.angle_argument()
                            if (target.text == "U" and len(args) < 4)
                            or (len(args) == 1 and target.text in {"R", "phase"})
                            else self.expression()
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
                if scope_depth:
                    reject(
                        operator, "scope", "control is admitted only at function scope"
                    )
                body = self.block(allow_scopes=False, scope_depth=scope_depth + 1)
                statements.append(Statement(operator, arguments, control_body=body))
            else:
                self.take(";")
                statements.append(Statement(operator, arguments))
        self.take("}")
        return tuple(statements)

    def parse(
        self,
    ) -> tuple[tuple[Function, ...], tuple[tuple[Token, Expression], ...]]:
        if self.current.text == "<eof>":
            self.take("qfunc")
        functions: list[Function] = []
        constants: list[tuple[Token, Expression]] = []
        while self.current.text != "<eof>":
            if len(functions) + len(constants) >= MAX_FUNCTIONS:
                reject(
                    self.current, "budget.functions", "source exceeds 32 definitions"
                )
            kind = (
                self.tokens[self.position + 2].text
                if self.position + 2 < len(self.tokens)
                else ""
            )
            if (
                self.current.text != "qfunc"
                and self.tokens[self.position + 1].text == ":"
                and kind in {"SparsePauliOp", "real"}
            ):
                name = self.name()
                self.take(":")
                self.take(kind)
                array = kind == "real" and self.current.text == "["
                if array:
                    self.take("[")
                    self.take("]")
                self.take("=")
                if kind == "SparsePauliOp":
                    value = self.sparse_pauli_op()
                elif array:
                    value = self.literal_array(
                        partial(self.preparation_literal, allow_fraction=False)
                    )
                else:
                    value = self.signed_angle_argument()
                constants.append((name, value))
                self.take(";")
            else:
                functions.append(self.function())
        self.take("<eof>")
        return tuple(functions), tuple(constants)
