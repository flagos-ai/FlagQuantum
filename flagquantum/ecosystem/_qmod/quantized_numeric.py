"""Bounded lexical numeric programs with a proved discard environment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

from .angles import REAL_LITERAL
from .masked_selection import emit_truth_xor
from .syntax import Expression, Function, Parameter, Statement, Token, reject

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Value
    from .context import _Adapt

# Qmod's default precision for the pinned native profile. Source numeric
# literals are represented at up to twenty bits before the profile truncation.
FRACTION_BITS = 8
_LITERAL_BITS = 20
_MAX_BRANCH_BITS = 5
_ROWS = 1 << _MAX_BRANCH_BITS


@dataclass(frozen=True)
class _Scalar:
    width: int
    signed: bool
    fraction: int
    low: Fraction
    high: Fraction
    values: tuple[Fraction, ...]


@dataclass(frozen=True)
class NumericProgram:
    bits: int
    outputs: tuple[_Scalar, ...]


def _truncate(value: Fraction, bits: int) -> Fraction:
    scale = 1 << bits
    return Fraction((value * scale).__floor__(), scale)


def _number(token: Token, value: Fraction) -> _Scalar:
    literal_scale = 1 << _LITERAL_BITS
    native = Fraction(int(value * literal_scale), literal_scale)
    quantized = _truncate(native, FRACTION_BITS)
    fraction = quantized.denominator.bit_length() - 1
    return _scalar(token, quantized, quantized, fraction, (quantized,) * _ROWS)


def _scalar(
    token: Token,
    low: Fraction,
    high: Fraction,
    fraction: int,
    values: tuple[Fraction, ...],
) -> _Scalar:
    fraction = min(fraction, FRACTION_BITS)
    scale = 1 << fraction
    lower, upper = int(low * scale), int(high * scale)
    signed = lower < 0
    if signed:
        width = max((~lower).bit_length(), max(upper, 0).bit_length()) + 1
    else:
        width = upper.bit_length()
    width = max(width, fraction, 1)
    if width > 20:
        reject(token, "budget.precision", "quantized output format exceeds twenty bits")
    if any(
        value * scale != int(value * scale) or value < low or value > high
        for value in values
    ):
        reject(token, "precision", "quantized result does not fit the inferred format")
    return _Scalar(width, signed, fraction, low, high, values)


def _literal(node: Expression) -> Fraction | None:
    if not node.children and REAL_LITERAL.fullmatch(node.token.text):
        number = float(node.token.text)
        if not math.isfinite(number) or abs(number) > 2**20:
            reject(
                node.token,
                "precision",
                "numeric literal exceeds the bounded fixed-point range",
            )
        return Fraction(number)
    if node.token.text == "-" and len(node.children) == 1:
        value = _literal(node.children[0])
        return None if value is None else -value
    return None


def _expression(node: Expression, values: Mapping[str, _Scalar]) -> _Scalar:
    literal = _literal(node)
    if literal is not None:
        return _number(node.token, literal)
    name, args = node.token.text, node.children
    if not args and name in values:
        return values[name]
    if name == "-" and len(args) == 1:
        operand = _expression(args[0], values)
        return _scalar(
            node.token,
            -operand.high,
            -operand.low,
            operand.fraction,
            tuple(-value for value in operand.values),
        )
    if name == "**" and len(args) == 2:
        base = _expression(args[0], values)
        degree = _literal(args[1])
        if degree is None or degree.denominator != 1 or not 1 <= degree <= 4:
            reject(
                node.token,
                "precision",
                "quantized power requires a literal degree in 1..4",
            )
        exponent = int(degree)
        fraction = min(base.fraction * exponent, FRACTION_BITS)
        edges = [base.low**exponent, base.high**exponent]
        if exponent % 2 == 0 and base.low < 0 < base.high:
            edges.append(Fraction())
        return _scalar(
            node.token,
            _truncate(min(edges), fraction),
            _truncate(max(edges), fraction),
            fraction,
            tuple(_truncate(value**exponent, fraction) for value in base.values),
        )
    if name in {"+", "-", "*"} and len(args) == 2:
        left, right = (_expression(child, values) for child in args)
        pairs = tuple(zip(left.values, right.values, strict=True))
        fraction = max(left.fraction, right.fraction)
        if name == "+":
            return _scalar(
                node.token,
                left.low + right.low,
                left.high + right.high,
                fraction,
                tuple(a + b for a, b in pairs),
            )
        if name == "-":
            return _scalar(
                node.token,
                left.low - right.high,
                left.high - right.low,
                fraction,
                tuple(a - b for a, b in pairs),
            )
        fraction = min(left.fraction + right.fraction, FRACTION_BITS)
        bounds = [
            a * b for a, b in product((left.low, left.high), (right.low, right.high))
        ]
        return _scalar(
            node.token,
            _truncate(min(bounds), fraction),
            _truncate(max(bounds), fraction),
            fraction,
            tuple(_truncate(a * b, fraction) for a, b in pairs),
        )
    reject(node.token, "precision", "unsupported lexical quantized numeric expression")


class _Planner:
    def __init__(self, definitions: Mapping[str, Function]) -> None:
        self.definitions = definitions
        self.bits = 0
        self.steps = 0
        self.used: set[str] = set()
        self.dropped: list[_Scalar] = []

    def fresh_uniform(self, token: Token) -> _Scalar:
        if self.bits == _MAX_BRANCH_BITS:
            reject(
                token,
                "budget.precision",
                "quantized preparation exceeds five independent basis bits",
            )
        bit = self.bits
        self.bits += 1
        return _scalar(
            token,
            Fraction(0),
            Fraction(1),
            0,
            tuple(Fraction(row >> bit & 1) for row in range(_ROWS)),
        )

    def function(
        self,
        syntax: Function,
        inputs: Mapping[str, _Scalar],
        active: tuple[str, ...],
    ) -> tuple[_Scalar, ...]:
        name = syntax.name.text
        if (
            name in active
            or len(active) >= 12
            or syntax.disable_const_checks
            or syntax.disable_permutation_check
        ):
            reject(
                syntax.name,
                "ownership",
                "quantized helpers must be bounded and acyclic",
            )
        self.used.add(name)
        roles = {port.name.text: port.role for port in syntax.parameters}
        if len(roles) != len(syntax.parameters) or any(
            port.kind != "qnum"
            or port.width is not None
            or port.record is not None
            or port.role not in {"output", "const", "inout"}
            for port in syntax.parameters
        ):
            reject(
                syntax.name, "type", "quantized helpers require inferred numeric ports"
            )
        initialized = dict(inputs)
        retired: set[str] = set()
        zero_allocations: set[str] = set()

        def reserve(node: Expression) -> str:
            target = node.token.text
            if (
                node.children
                or roles.get(target) not in {"local", "output"}
                or target in initialized
                or target in retired
            ):
                reject(
                    node.token,
                    "ownership",
                    "quantized assignment requires a fresh local or Output",
                )
            return target

        def read(node: Expression) -> _Scalar:
            if node.children or node.token.text not in initialized:
                reject(
                    node.token,
                    "ownership",
                    "quantized call requires one live numeric variable",
                )
            return initialized[node.token.text]

        for statement in syntax.statements:
            self.steps += 1
            token, args = statement.token, statement.arguments
            if self.steps > 256:
                reject(
                    token,
                    "budget.precision",
                    "quantized helper expansion exceeds 256 statements",
                )
            if statement.declaration is not None:
                parameter = statement.declaration
                if (
                    parameter.kind != "qnum"
                    or parameter.width is not None
                    or parameter.name.text in roles
                ):
                    reject(
                        parameter.name,
                        "type",
                        "quantized local must be a fresh inferred qnum",
                    )
                roles[parameter.name.text] = "local"
                continue
            if statement != Statement(token, args):
                reject(
                    token,
                    "scope",
                    "quantized numeric evaluation requires straight-line ownership",
                )
            if syntax.permutation and token.text != "=":
                reject(
                    token,
                    "permutation",
                    "quantized qperm may only compute fresh numeric results",
                )
            if token.text == "=" and len(args) == 2:
                target = reserve(args[0])
                initialized[target] = _expression(args[1], initialized)
            elif token.text == "allocate" and len(args) == 2 and _literal(args[0]) == 1:
                target = reserve(args[1])
                initialized[target] = _number(token, Fraction())
                zero_allocations.add(target)
            elif token.text == "hadamard_transform" and len(args) == 1:
                target = args[0].token.text
                if args[0].children or target not in zero_allocations:
                    reject(
                        token,
                        "ownership",
                        "quantized Hadamard needs a fresh one-bit allocation",
                    )
                zero_allocations.remove(target)
                initialized[target] = self.fresh_uniform(token)
            elif token.text == "prepare_state" and len(args) == 3:
                vector, bound, destination = args
                if (
                    _literal(bound) != 0
                    or vector.token.text != "["
                    or tuple(map(_literal, vector.children))
                    != (Fraction(1, 2), Fraction(1, 2))
                ):
                    reject(
                        token,
                        "amplitudes",
                        "quantized preparation needs an exact uniform bit",
                    )
                initialized[reserve(destination)] = self.fresh_uniform(token)
            elif token.text == "drop" and len(args) == 1:
                if roles.get(args[0].token.text) != "local":
                    reject(token, "ownership", "quantized drop requires an owned local")
                self.dropped.append(read(args[0]))
                retired.add(args[0].token.text)
                initialized.pop(args[0].token.text)
            elif token.text in self.definitions:
                helper = self.definitions[token.text]
                if len(args) != len(helper.parameters) or len(
                    {arg.token.text for arg in args}
                ) != len(args):
                    reject(
                        token,
                        "alias",
                        "quantized helper actuals must be disjoint whole words",
                    )
                child_inputs = {}
                child_outputs = []
                for port, actual in zip(helper.parameters, args, strict=True):
                    if port.role == "output":
                        child_outputs.append(reserve(actual))
                    else:
                        child_inputs[port.name.text] = read(actual)
                result = self.function(helper, child_inputs, (*active, name))
                initialized.update(zip(child_outputs, result, strict=True))
            else:
                reject(
                    token, "scope", "unsupported effect in quantized numeric program"
                )
        results = []
        for port in syntax.parameters:
            if port.role != "output":
                continue
            value = initialized.get(port.name.text)
            if value is None:
                reject(
                    port.name,
                    "ownership",
                    "quantized helper must initialize its Outputs",
                )
            results.append(value)
        return tuple(results)


def try_numeric_program(definitions: Mapping[str, Function]) -> NumericProgram | None:
    main = definitions["main"]
    if (
        len(main.parameters) < 2
        or any(
            port.role != "output" or port.kind != "qnum" or port.width is not None
            for port in main.parameters
        )
        or not any(statement.token.text == "drop" for statement in main.statements)
    ):
        return None

    def non_dyadic(node: Expression) -> bool:
        literal = _literal(node)
        if literal is not None:
            return literal.denominator > 1 << _LITERAL_BITS
        return any(non_dyadic(child) for child in node.children)

    if not any(
        function.permutation
        and any(
            port.role == "const" and port.kind == "qnum" and port.width is None
            for port in function.parameters
        )
        and any(
            statement.token.text == "=" and non_dyadic(statement.arguments[-1])
            for statement in function.statements
        )
        for function in definitions.values()
    ):
        return None
    planner = _Planner(definitions)
    outputs = planner.function(main, {}, ())
    if planner.used != definitions.keys() or not planner.bits:
        reject(
            main.name,
            "ownership",
            "quantized profile requires one closed reachable program",
        )
    rows = 1 << planner.bits
    labels = {
        tuple(item.values[row] for item in planner.dropped) for row in range(rows)
    }
    if len(labels) != rows:
        reject(
            main.name,
            "ownership",
            "dropped numeric state must distinguish every prepared branch",
        )
    return NumericProgram(planner.bits, outputs)


def emit_numeric_program(owner: _Adapt, program: NumericProgram) -> None:
    token = owner.syntax.name
    branch = Token("$quantized_branch", token.location)
    owner.parameters[branch.text] = Parameter(branch, "qbit[]", program.bits, "local")
    owner.widths[branch.text] = program.bits
    selector = owner.register(branch, allow_qubit_array=True, allocate_local=True)
    owner.effect_op("qreg.allocate", token, (selector,))
    reads = tuple((selector, bit) for bit in range(program.bits))
    for _, bit in reads:
        owner.effect_op("qreg.gate", token, (selector,), index=bit, gate="h")
    shared: dict[tuple[int, ...], tuple[Value, int]] = {}
    for port, output in zip(owner.syntax.parameters, program.outputs, strict=True):
        target = owner.register(port.name, output.width, allow_fractional_output=True)
        if output.fraction:
            owner.fractional_bits[port.name.text] = output.fraction
        if output.signed:
            owner.signed.add(port.name.text)
        owner.effect_op("qreg.allocate", token, (target,))
        raw = [
            int(value * (1 << output.fraction))
            for value in output.values[: 1 << program.bits]
        ]
        for bit in range(output.width):
            table = tuple((value >> bit) & 1 for value in raw)
            if not any(table):
                continue
            same = shared.get(table)
            inverse = shared.get(tuple(1 - value for value in table))
            if inverse is not None:
                owner.effect_op("qreg.gate", token, (target,), gate="x", index=bit)
            if (source := same or inverse) is not None:
                owner.emit_indexed_x(token, (source, (target, bit)))
            else:
                emit_truth_xor(owner, token, reads, (target, bit), list(table))
                shared[table] = (target, bit)
    owner.effect_op("qreg.drop", token, (selector,))
