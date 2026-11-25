"""Adapter-local bounded syntax, discarded at the Compiler model boundary."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Literal

from .angles import SCIENTIFIC_LITERAL, parse_literal

if TYPE_CHECKING:
    pass

from .expression_parser import expression as _expression
from .statement_parser import block as _block
from .syntax_model import (
    _COMPARISONS as _COMPARISONS,
)
from .syntax_model import (
    _RESERVED as _RESERVED,
)
from .syntax_model import (
    _TOKEN as _TOKEN,
)
from .syntax_model import (
    ANGLE_CALLS as ANGLE_CALLS,
)
from .syntax_model import (
    EXTREMA as EXTREMA,
)
from .syntax_model import (
    MAX_BYTES as MAX_BYTES,
)
from .syntax_model import (
    MAX_DEPTH as MAX_DEPTH,
)
from .syntax_model import (
    MAX_EXPRESSION_DEPTH as MAX_EXPRESSION_DEPTH,
)
from .syntax_model import (
    MAX_FUNCTIONS as MAX_FUNCTIONS,
)
from .syntax_model import (
    MAX_PHASE_EXPRESSION_DEPTH as MAX_PHASE_EXPRESSION_DEPTH,
)
from .syntax_model import (
    MAX_SCOPE_DEPTH as MAX_SCOPE_DEPTH,
)
from .syntax_model import (
    MAX_TOKENS as MAX_TOKENS,
)
from .syntax_model import (
    PREPARATIONS as PREPARATIONS,
)
from .syntax_model import (
    Expression as Expression,
)
from .syntax_model import (
    Function as Function,
)
from .syntax_model import (
    Parameter as Parameter,
)
from .syntax_model import (
    Record as Record,
)
from .syntax_model import (
    Statement as Statement,
)
from .syntax_model import (
    Token as Token,
)
from .syntax_model import (
    exact_unsigned_literal as exact_unsigned_literal,
)
from .syntax_model import (
    reject as reject,
)
from .syntax_model import (
    tokenize as tokenize,
)


class Parser:
    """Bounded source definitions; records stay within this private adapter."""

    def __init__(self, tokens: tuple[Token, ...]) -> None:
        self.tokens = tokens
        self.position = 0
        self.closed_fixed_point = False
        self.zero_walk_proof = False
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

    expression = _expression

    def binary_log_width(
        self,
        *,
        symbolic: bool = False,
        nesting: int = 0,
        depth_limit: int = MAX_EXPRESSION_DEPTH,
    ) -> Expression:
        token = self.take(self.current.text)
        self.take("(")
        self.take("log")
        self.take("(")
        if symbolic:
            operand = self.expression(
                nesting=nesting + 1, static=True, depth_limit=depth_limit
            )
            if operand.depth >= depth_limit:
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

    def preparation_literal(
        self, *, allow_fraction: bool, allow_sqrt: bool = True
    ) -> Expression:
        grouped = self.current.text == "("
        if grouped:
            self.take("(")
        numerator = self.current
        if numerator.text == "sqrt":
            if not allow_sqrt:
                reject(
                    numerator, "amplitudes", "square root requires a literal argument"
                )
            self.take("sqrt")
            self.take("(")
            argument = self.preparation_literal(allow_fraction=True, allow_sqrt=False)
            self.take(")")
            value = Expression(numerator, (argument,))
        elif (
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
        if self.current.text == ".":
            self.take(".")
        else:
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
        if not token.text.isdecimal() and not SCIENTIFIC_LITERAL.fullmatch(token.text):
            reject(token, "matrix", "expected a numeric matrix literal")
        if (
            SCIENTIFIC_LITERAL.fullmatch(token.text)
            or self.tokens[self.position + 1].text == "."
        ):
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
            "IndexedPauli", {"pauli": self.sparse_axis, "index": self.expression}
        )

    def sparse_axis(self) -> Expression:
        token = self.current
        if token.text in {"0", "1", "2", "3"}:
            self.take(token.text)
            return Expression(replace(token, text="IXYZ"[int(token.text)]))
        return self.pauli_axis()

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
    ) -> tuple[str, int | None, int]:
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

                phase_kind, fractional_bits = parse_fraction(
                    self, width, allow_phase, allow_signed_integer=allow_signed_integer
                )
                if phase_kind is not None:
                    return phase_kind, width, fractional_bits
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
            return "qbit[]", None, 0
        if kind.text == "qbit" and self.current.text == "[" and allow_fixed_array:
            self.take("[")
            width = self.width()
            self.take("]")
            return "qbit[]", width, 0
        if self.current.text == "[" and kind.text == "qnum" and allow_numeric_array:
            return kind.text, width, 0
        if self.current.text == "[":
            reject(
                self.current,
                "type",
                "arrays, slices and bind require separate view semantics",
            )
        return kind.text, width, 0

    def function(self) -> Function:
        annotation = None
        disable_const = None
        while self.current.text == "@":
            marker = self.take("@")
            if self.current.text == "disable_perm_check" and annotation is None:
                annotation = self.take("disable_perm_check")
            elif self.current.text == "disable_const_checks" and disable_const is None:
                self.take("disable_const_checks")
                self.take("(")
                disable_const = self.name()
                self.take(")")
            else:
                reject(marker, "annotation", "unknown or repeated Qmod annotation")
        permutation = self.current.text == "qperm"
        if annotation is not None and not permutation:
            reject(annotation, "permutation", "permutation annotation requires qperm")
        self.take("qperm" if permutation else "qfunc")
        name = self.name()
        self.take("(")
        parameters: list[Parameter] = []
        while self.current.text != ")":
            role: Literal[
                "output",
                "const",
                "real",
                "integer",
                "inout",
                "callable",
                "operator",
                "input",
            ]
            if self.current.text == "input" and permutation and name.text != "main":
                self.take("input")
                role = "input"
            elif self.current.text == "output":
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
            ] in ([":", "real"], [":", "int"], [":", "Pauli"]) or (
                self.zero_walk_proof
                and self.tokens[self.position + 2].text == "bool"
                and name.text != "main"
            ):
                role = (
                    "real"
                    if self.tokens[self.position + 2].text == "real"
                    else "integer"
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
            ] in ([":", "SparsePauliOp"], [":", "PauliTerm"], [":", "IndexedPauli"]):
                role = "operator"
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
            fractional_bits = 0
            numeric_length = None
            if self.current.text in self.records:
                if role not in {"output", "inout", "const"}:
                    reject(
                        self.current,
                        "role",
                        "records require Output or writable helper ports",
                    )
                record = self.take(self.current.text).text
                kind, width = "qbit[]", self.records[record].width
            elif role == "operator":
                kind, width = self.take(self.current.text).text, None
                if kind in {"PauliTerm", "IndexedPauli"}:
                    self.take("[")
                    self.take("]")
                    kind += "[]"
            elif role == "callable":
                from .quantum_callbacks import parse_signature

                kind, width = parse_signature(self), None
            elif role in {"real", "integer"}:
                kind = self.current.text
                self.take(kind)
                width = None
                if self.current.text == "[":
                    self.take("[")
                    if name.text == "main" or self.current.text != "]":
                        width = self.width()
                    self.take("]")
                    kind += "[]"
                    if (
                        kind == "real[]"
                        and width is not None
                        and name.text != "main"
                        and self.current.text == "["
                    ):
                        self.take("[")
                        self.take("]")
                        kind = "real[][]"
                if kind.startswith("Pauli") and (
                    kind != "Pauli[]" or name.text == "main"
                ):
                    reject(
                        parameter,
                        "pauli",
                        "Pauli arrays are admitted as static helper ports",
                    )
            else:
                kind, width, fractional_bits = self.scalar_type(
                    allow_qubit_array=name.text == "main"
                    or role in {"inout", "const"}
                    or permutation
                    and role == "output",
                    allow_phase=True,
                    allow_signed_integer=role in {"const", "inout"}
                    or name.text == "main"
                    and role == "output",
                    allow_fixed_array=role in {"inout", "output", "const", "input"}
                    or name.text == "main",
                    allow_numeric_array=role in {"inout", "output", "const"},
                )
                if kind == "qnum" and self.current.text == "[":
                    self.take("[")
                    if self.zero_walk_proof and self.current.text == "]":
                        numeric_length = None
                    else:
                        numeric_length = self.width()
                    self.take("]")
                    if (self.closed_fixed_point and width is None) or (
                        self.zero_walk_proof and numeric_length is None
                    ):
                        kind = "qnum[]"
                    elif (
                        width is None
                        or numeric_length is None
                        or width * numeric_length > 256
                    ):
                        reject(
                            parameter,
                            "width",
                            "numeric array requires a fixed element width within 256 total bits",
                        )
                    else:
                        width *= numeric_length
                        kind = "qbit[]"
                if kind == "qphase_signed" and name.text != "main":
                    reject(
                        parameter,
                        "type",
                        "signed fixed-point helper ports are not admitted",
                    )
            parameters.append(
                Parameter(
                    parameter,
                    kind,
                    width,
                    role,
                    record,
                    length=numeric_length,
                    fractional_bits=fractional_bits,
                )
            )
            if self.current.text != ",":
                break
            self.take(",")
            if self.current.text == ")":
                reject(
                    self.current, "syntax", "trailing parameter commas are not admitted"
                )
        self.take(")")
        return Function(
            name,
            tuple(parameters),
            self.block(),
            permutation=permutation,
            disable_permutation_check=annotation,
            disable_const_checks=disable_const,
        )

    def call_argument(self) -> Expression:
        if self.current.text == "lambda":
            from .quantum_callbacks import parse_lambda

            return parse_lambda(self)
        if self.current.text == "[":
            if self.tokens[self.position + 1].text == "[":

                def row() -> Expression:
                    items = self.literal_array(self.real_angle)
                    return replace(items, token=replace(items.token, text="[]"))

                matrix = self.literal_array(row)
                return replace(matrix, token=replace(matrix.token, text="$real_matrix"))
            if self.tokens[self.position + 1].text == "IndexedPauli":
                paulis = self.literal_array(self.indexed_pauli)
                return replace(
                    paulis, token=replace(paulis.token, text="IndexedPauli[]")
                )
            if self.tokens[self.position + 1].text == "PauliTerm":
                terms = self.pauli_terms()
                return replace(terms, token=replace(terms.token, text="PauliTerm[]"))
            if self.tokens[self.position + 1].text == "Pauli":
                array = self.literal_array(self.pauli_axis)
                return Expression(
                    Token("$pauli_axes", array.token.location), array.children
                )
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
            items = [self.expression(allow_slice=True)]
            while self.current.text == ",":
                self.take(",")
                items.append(self.expression(allow_slice=True))
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

    block = _block

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
                nested = self.name()
                self.take(";")
                fields.append(Parameter(field, "qbit[]", None, record=nested.text))
                continue
            kind, width, fractional_bits = self.scalar_type(
                allow_fixed_array=True,
                allow_numeric_array=True,
                allow_phase=True,
                allow_signed_integer=True,
            )
            if (
                kind == "qphase_signed"
                or (kind == "qphase" or fractional_bits)
                and self.current.text == "["
            ):
                reject(
                    field,
                    "type",
                    "fractional record fields require fixed-point field lowering",
                )
            if width is None and (kind != "qnum" or self.current.text == "["):
                reject(
                    field,
                    "width",
                    "record numeric arrays require a fixed element width",
                )
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
            fields.append(
                Parameter(
                    field, kind, width, length=length, fractional_bits=fractional_bits
                )
            )
            if sum((item.width or 0) * (item.length or 1) for item in fields) > 256:
                reject(field, "budget.record", "record exceeds 256 bits")
        self.take("}")
        if not fields:
            reject(name, "type", "record requires at least one field")
        self.records[name.text] = Record(name, tuple(fields))

    def resolve_records(self) -> None:
        def resolve(name: str, active: tuple[str, ...]) -> int:
            layout = self.records[name]
            if all(field.record is None for field in layout.fields):
                return layout.width
            if name in active or len(active) >= MAX_SCOPE_DEPTH:
                reject(
                    layout.name,
                    "record.cycle",
                    "record fields must form a bounded acyclic layout",
                )
            fields = []
            for field in layout.fields:
                if field.record is not None:
                    if field.record not in self.records:
                        reject(
                            field.name,
                            "record.type",
                            f"unknown field record {field.record!r}",
                        )
                    field = replace(field, width=resolve(field.record, (*active, name)))
                fields.append(field)
            layout = replace(layout, fields=tuple(fields))
            if layout.width > 256:
                reject(layout.name, "budget.record", "record exceeds 256 bits")
            self.records[name] = layout
            return layout.width

        for name in self.records:
            resolve(name, ())

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
                    self.current,
                    "budget.functions",
                    f"source exceeds {MAX_FUNCTIONS} definitions",
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
                self.resolve_records()
                functions.append(self.function())
        from .record_inference import bind_record_widths, infer_prepared_record_widths

        parsed = tuple(functions)
        changed = infer_prepared_record_widths(self.records, parsed)
        self.resolve_records()
        if changed:
            parsed = bind_record_widths(self.records, parsed)
        self.take("<eof>")
        return parsed, tuple(constants)
