"""Specialize a terminal reversible helper that only reads a local fixed-point value."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from operator import add, eq, ge, gt, le, lt, mul, ne, sub, truediv
from typing import TYPE_CHECKING

from .angles import REAL_LITERAL
from .computed_rotation import exact_expression
from .masked_selection import emit_selected, emit_truth_xor
from .preparation import exponential_angles
from .syntax import Expression, Function, Statement, Token, reject

if TYPE_CHECKING:
    from ...compiler._hybrid.model import Operation
    from .context import _Adapt
    from .phase import Location

Number = Fraction | float

_COMPARISONS: dict[str, Callable[[Number, Number], bool]] = {
    "==": eq,
    "!=": ne,
    ">": gt,
    ">=": ge,
    "<": lt,
    "<=": le,
}
_BINARY: dict[str, Callable[[Number, Number], Number]] = {
    "+": add,
    "-": sub,
    "*": mul,
    "/": truediv,
    "max": max,
    "min": min,
}
_EXACT_ARITHMETIC: dict[str, Callable[[Fraction, Fraction], Fraction]] = {
    "+": add,
    "-": sub,
    "*": mul,
}
_UNARY: dict[str, Callable[[Number], Number]] = {
    "-": lambda value: -value,
    "acos": math.acos,
    "asin": math.asin,
    "sqrt": math.sqrt,
    "exp": math.exp,
}


@dataclass(frozen=True)
class Word:
    width: int
    fraction: int = 0
    bits: int | None = None
    wires: tuple[Location, ...] = ()

    @property
    def value(self) -> Fraction:
        assert self.bits is not None
        return Fraction(self.bits, 1 << self.fraction)


Binding = Word | int | float


def _atom(node: Expression, name: str) -> bool:
    return not node.children and node.token.text == name


def _plain(statement: Statement) -> bool:
    return statement == Statement(statement.token, statement.arguments)


def inferred_computation(
    function: Function,
) -> tuple[tuple[str, ...], Expression] | None:
    if (
        not function.permutation
        or function.disable_const_checks
        or function.disable_permutation_check
        or len(function.statements) != 1
        or not function.parameters
        or not any(port.width is None for port in function.parameters[:-1])
        or any(
            port.kind != "qnum" or port.record is not None or port.fractional_bits
            for port in function.parameters
        )
        or tuple(p.role for p in function.parameters)
        != (*("const" for _ in function.parameters[:-1]), "output")
        or function.parameters[-1].width is not None
    ):
        return None
    assignment = function.statements[0]
    if (
        not _plain(assignment)
        or assignment.token.text != "="
        or len(assignment.arguments) != 2
        or not _atom(assignment.arguments[0], function.parameters[-1].name.text)
    ):
        return None
    return tuple(p.name.text for p in function.parameters[:-1]), assignment.arguments[1]


def _nested(statements: tuple[Statement, ...]) -> Iterator[Statement]:
    for statement in statements:
        yield statement
        for scope in (
            statement.within,
            statement.apply,
            statement.repeat_body,
            statement.control_body,
            statement.control_else,
            statement.inverse_body,
            statement.static_if_body,
        ):
            if scope:
                yield from _nested(scope)


def payoff_helpers(definitions: Mapping[str, Function]) -> frozenset[str]:
    """Find helpers whose unsized numeric bindings are checked at payoff call sites."""
    helpers: set[str] = set()

    def visit(name: str) -> None:
        if name in helpers or name not in definitions or name == "main":
            return
        helpers.add(name)
        for stmt in _nested(definitions[name].statements):
            visit(stmt.token.text)

    for function in definitions.values():
        if len(function.statements) < 3:
            continue
        declaration, compute, apply = function.statements[-3:]
        local = declaration.declaration
        callee = definitions.get(compute.token.text)
        consumer = definitions.get(apply.token.text)
        if (
            local is not None
            and local.kind == "qnum"
            and local.width is None
            and callee is not None
            and consumer is not None
            and inferred_computation(callee) is not None
            and compute.arguments
            and _atom(compute.arguments[-1], local.name.text)
            and any(_atom(arg, local.name.text) for arg in apply.arguments)
        ):
            visit(callee.name.text)
            visit(consumer.name.text)
    return frozenset(helpers)


@dataclass(frozen=True)
class _Shape:
    low: Fraction
    high: Fraction
    fraction: int


def _checked_shape(
    node: Expression, low: Fraction, high: Fraction, fraction: int
) -> _Shape:
    if fraction > 5 or max(abs(low), abs(high)) >= 64:
        reject(
            node.token,
            "precision",
            "fixed-point payoff intermediates exceed the bounded format",
        )
    return _Shape(low, high, fraction)


def _layout(node: Expression, widths: dict[str, int]) -> _Shape:
    name, args = node.token.text, node.children
    if not args:
        if name in widths:
            return _Shape(Fraction(0), Fraction((1 << widths[name]) - 1), 0)
        if REAL_LITERAL.fullmatch(name):
            value = Fraction(name)
            if value.denominator & (value.denominator - 1):
                reject(
                    node.token,
                    "precision",
                    "inferred payoff requires exact dyadic coefficients",
                )
            return _checked_shape(
                node, value, value, value.denominator.bit_length() - 1
            )
    if name == "-" and len(args) == 1:
        shape = _layout(args[0], widths)
        return _Shape(-shape.high, -shape.low, shape.fraction)
    if name in {"+", "-", "*", "min", "max"} and len(args) == 2:
        left, right = (_layout(child, widths) for child in args)
        # The extrema share a promoted representation that can store either operand.
        if name in {"min", "max"}:
            return _Shape(
                min(left.low, right.low),
                max(left.high, right.high),
                max(left.fraction, right.fraction),
            )
        operation = _EXACT_ARITHMETIC[name]
        values = [
            operation(a, b)
            for a, b in product((left.low, left.high), (right.low, right.high))
        ]
        fraction = (
            left.fraction + right.fraction
            if name == "*"
            else max(left.fraction, right.fraction)
        )
        return _checked_shape(node, min(values), max(values), fraction)
    reject(node.token, "precision", "unsupported inferred payoff arithmetic")


class _Payoff:
    def __init__(self, owner: _Adapt, selector: tuple[Location, ...]) -> None:
        self.owner = owner
        self.selector = selector
        self.steps = 0

    def word(self, node: Expression, env: dict[str, Binding]) -> Word:
        value = env.get(node.token.text) if not node.children else None
        if isinstance(value, Word):
            return value
        if node.token.text == "[":
            array, index_expr = node.children
            parent = self.word(array, env)
            index = self.integer(index_expr, env)
            if not 0 <= index < parent.width:
                reject(index_expr.token, "width", "payoff bit index is out of bounds")
            if parent.bits is not None:
                return Word(1, bits=(parent.bits >> index) & 1)
            return Word(1, wires=(parent.wires[index],))
        reject(node.token, "ownership", "payoff may only use its lexical quantum ports")

    def number(self, node: Expression, env: dict[str, Binding]) -> Fraction | float:
        text, args = node.token.text, node.children
        if not args:
            value = env.get(text)
            if isinstance(value, (int, float)):
                return value if isinstance(value, float) else Fraction(value)
            if value is None and text == "pi":
                return math.pi
            if REAL_LITERAL.fullmatch(text):
                return Fraction(text)
        if text == "." and len(args) == 2:
            word = self.word(args[0], env)
            prop = args[1].token.text
            if prop in {"size", "len", "fraction_digits"}:
                return Fraction(
                    word.fraction if prop == "fraction_digits" else word.width
                )
        if len(args) == 1 and text in _UNARY:
            return _UNARY[text](self.number(args[0], env))
        if len(args) == 2:
            if text in _BINARY:
                a, b = (self.number(child, env) for child in args)
                return _BINARY[text](a, b)
            if text == "**":
                exponent = self.integer(args[1], env)
                if abs(exponent) > 32:
                    reject(node.token, "budget", "payoff exponent exceeds 32")
                return self.number(args[0], env) ** exponent
        word = self.word(node, env)
        if word.bits is None:
            reject(
                node.token,
                "ownership",
                "payoff angle or static guard cannot read a writable port",
            )
        return word.value

    def integer(self, node: Expression, env: dict[str, Binding]) -> int:
        value = self.number(node, env)
        if value != int(value):
            reject(node.token, "type", "payoff requires an exact integer")
        return int(value)

    def condition(self, node: Expression, env: dict[str, Binding]) -> bool:
        if node.token.text not in _COMPARISONS:
            return bool(self.number(node, env))
        left, right = (self.number(arg, env) for arg in node.children)
        return _COMPARISONS[node.token.text](left, right)

    def gate(self, token: Token, gate: str, word: Word, **attributes: object) -> None:
        if not word.wires or len(word.wires) != 1 or word.wires[0] in self.selector:
            reject(token, "alias", "payoff needs a disjoint writable qubit")
        value, bit = word.wires[0]
        self.owner.effect_op(
            "qreg.gate", token, (value,), index=bit, gate=gate, **attributes
        )

    def predicate(self, statement: Statement, env: dict[str, Binding]) -> None:
        target_arg, condition = statement.arguments
        target = self.word(target_arg, env)
        if len(target.wires) != 1 or target.wires[0] in self.selector:
            reject(
                target_arg.token,
                "alias",
                "payoff predicate requires a distinct writable target",
            )

        # Writable numeric ports in an XOR predicate are read as coherent selectors.
        def reads(node: Expression) -> set[str]:
            name = node.token.text
            if not node.children:
                item = env.get(name)
                return {name} if isinstance(item, Word) and item.wires else set()
            return set().union(*(reads(arg) for arg in node.children))

        inputs = []
        for name in sorted(reads(condition)):
            source = env[name]
            assert isinstance(source, Word)
            inputs.append((name, source))
        wires = tuple(wire for _, word in inputs for wire in word.wires)
        if len(wires) > 5 or len(set(wires)) != len(wires) or target.wires[0] in wires:
            reject(
                condition.token,
                "alias",
                "payoff predicate needs up to five disjoint read bits",
            )
        if not wires:
            if self.condition(condition, env):
                self.gate(statement.token, "x", target)
            return
        table: list[int] = []
        for pattern in range(1 << len(wires)):
            bound = env.copy()
            offset = 0
            for name, word in inputs:
                bound[name] = replace(
                    word, bits=(pattern >> offset) & ((1 << word.width) - 1), wires=()
                )
                offset += word.width
            table.append(int(self.condition(condition, bound)))
        emit_truth_xor(self.owner, statement.token, wires, target.wires[0], table)

    def call(
        self, function: Function, args: tuple[Binding, ...], active: tuple[str, ...]
    ) -> None:
        if (
            function.permutation
            or function.disable_const_checks
            or function.disable_permutation_check
            or function.name.text in active
            or len(active) >= 8
            or len(args) != len(function.parameters)
        ):
            reject(
                function.name,
                "ownership",
                "payoff requires bounded reversible helper calls",
            )
        env: dict[str, Binding] = {}
        used: set[Location] = set()
        for port, arg in zip(function.parameters, args, strict=True):
            name = port.name.text
            if name in env:
                reject(port.name, "name", "duplicate payoff helper port")
            if isinstance(arg, Word):
                if (
                    port.kind not in {"qbit", "qbit[]", "qnum"}
                    or port.record is not None
                    or port.length is not None
                    or port.role not in {"const", "inout"}
                    or (port.role == "const") != (arg.bits is not None)
                    or (port.width is not None and port.width != arg.width)
                    or (port.fractional_bits and port.fractional_bits != arg.fraction)
                    or used.intersection(arg.wires)
                ):
                    reject(
                        port.name,
                        "ownership",
                        "payoff helper has an incompatible or aliased port",
                    )
                if port.kind == "qbit[]":
                    arg = replace(arg, fraction=0)
                used.update(arg.wires)
            elif port.role not in {"real", "integer"} or port.kind not in {
                "real",
                "int",
            }:
                reject(
                    port.name,
                    "signature",
                    "payoff helper needs a scalar classical binding",
                )
            env[name] = arg
        self.body(function.statements, env, (*active, function.name.text), set())

    def body(
        self,
        statements: tuple[Statement, ...],
        env: dict[str, Binding],
        active: tuple[str, ...],
        locals_: set[str],
    ) -> None:
        for statement in statements:
            self.steps += 1
            token, args = statement.token, statement.arguments
            if self.steps > 4096:
                reject(token, "budget", "fixed-point payoff exceeds expansion budget")
            if statement.control_body is not None:
                chosen = (
                    statement.control_body
                    if self.condition(args[0], env)
                    else statement.control_else
                )
                self.body(chosen or (), env.copy(), active, locals_)
            elif statement.repeat_index is not None:
                count = self.integer(args[0], env)
                index = statement.repeat_index.text
                if not 0 <= count <= 16 or index in env or index in locals_:
                    reject(
                        token,
                        "budget",
                        "payoff loop requires a fresh index and up to 16 iterations",
                    )
                for iteration in range(count):
                    self.body(
                        statement.repeat_body,
                        {**env, index: iteration},
                        active,
                        locals_,
                    )
            elif statement.declaration is not None:
                port = statement.declaration
                name = port.name.text
                if (
                    port.kind != "qnum"
                    or port.width is not None
                    or name in env
                    or name in locals_
                ):
                    reject(
                        token,
                        "ownership",
                        "payoff local must be an unallocated numeric alias",
                    )
                locals_.add(name)
            elif statement.within is not None:
                if len(statement.within) != 1:
                    reject(
                        token,
                        "ownership",
                        "payoff within needs one static reversible view transformation",
                    )
                step = statement.within[0]
                bound = env.copy()
                if (
                    _plain(step)
                    and step.token.text == "apply_to_all"
                    and len(step.arguments) == 2
                    and _atom(step.arguments[0], "X")
                ):
                    arg = step.arguments[1]
                    word = self.word(arg, env)
                    if arg.children or word.bits is None:
                        reject(
                            arg.token,
                            "ownership",
                            "payoff can invert only its local computed bits",
                        )
                    bound[arg.token.text] = replace(
                        word, bits=word.bits ^ ((1 << word.width) - 1)
                    )
                elif (
                    _plain(step)
                    and step.token.text == "->"
                    and len(step.arguments) == 2
                ):
                    src, dst = step.arguments
                    if src.children or dst.children or dst.token.text not in locals_:
                        reject(
                            token,
                            "ownership",
                            "payoff binding requires a fresh local view",
                        )
                    alias = self.word(src, env)
                    bound.pop(src.token.text)
                    bound[dst.token.text] = alias
                else:
                    reject(
                        token,
                        "ownership",
                        "payoff cannot change borrowed bits in within",
                    )
                self.body(statement.apply, bound, active, locals_)
            elif not _plain(statement):
                reject(token, "ownership", "unsupported scope in fixed-point payoff")
            elif token.text == "RY" and len(args) == 2:
                angle = float(self.number(args[0], env))
                if not math.isfinite(angle) or abs(angle) > 2**53:
                    reject(token, "angle", "payoff angle must be finite and bounded")
                self.gate(token, "ry", self.word(args[1], env), angle=angle)
            elif token.text == "prepare_exponential_state" and len(args) == 2:
                rate = self.number(args[0], env)
                word = self.word(args[1], env)
                if not word.wires:
                    reject(
                        token,
                        "ownership",
                        "exponential preparation needs writable storage",
                    )
                for wire, angle in zip(
                    word.wires, exponential_angles(float(rate), word.width), strict=True
                ):
                    self.gate(token, "ry", Word(1, wires=(wire,)), angle=angle)
            elif token.text == "^=" and len(args) == 2:
                self.predicate(statement, env)
            elif token.text in self.owner.definitions:
                callee = self.owner.definitions[token.text]
                if len(args) != len(callee.parameters):
                    reject(token, "signature", "payoff helper argument count disagrees")
                bound_args = tuple(
                    (
                        float(self.number(arg, env))
                        if p.role == "real"
                        else (
                            self.integer(arg, env)
                            if p.role == "integer"
                            else self.word(arg, env)
                        )
                    )
                    for p, arg in zip(callee.parameters, args, strict=True)
                )
                self.call(callee, bound_args, active)
            else:
                reject(
                    token,
                    "ownership",
                    "fixed-point payoff may only act reversibly on borrowed ports",
                )


def lower_fixed_payoff(owner: _Adapt, statements: tuple[Statement, ...]) -> bool:
    if len(statements) != 3:
        return False
    declaration, compute, apply = statements
    local = declaration.declaration
    function = owner.definitions.get(compute.token.text)
    consumer = owner.definitions.get(apply.token.text)
    if (
        local is None
        or local.kind != "qnum"
        or local.width is not None
        or local.name.text in owner.parameters
        or function is None
        or consumer is None
        or not _plain(compute)
        or not _plain(apply)
        or len(compute.arguments) != len(function.parameters)
        or len(apply.arguments) != len(consumer.parameters)
    ):
        return False
    inferred = inferred_computation(function)
    if inferred is None or not _atom(compute.arguments[-1], local.name.text):
        return False
    if sum(_atom(arg, local.name.text) for arg in apply.arguments) != 1:
        reject(
            apply.token,
            "ownership",
            "computed payoff value must have one read-only consumer",
        )
    names, expr = inferred
    views_list = []
    for port, arg in zip(function.parameters[:-1], compute.arguments[:-1], strict=True):
        field = owner.record_field(arg) if arg.token.text == "." else None
        parameter = (
            field[0]
            if field is not None
            else owner.parameters.get(arg.token.text) if not arg.children else None
        )
        view = owner.array_view(arg)
        if (
            parameter is None
            or parameter.kind != "qnum"
            or parameter.fractional_bits
            or parameter.length is not None
            or (port.width is not None and len(view) != port.width)
        ):
            reject(
                arg.token,
                "type",
                "fixed-point payoff needs unsigned integral source words",
            )
        views_list.append(view)
    views = tuple(views_list)
    selector = tuple(wire for view in views for wire in view)
    if not 1 <= len(selector) <= 4 or len(set(selector)) != len(selector):
        reject(
            compute.token,
            "budget",
            "fixed-point payoff requires up to four disjoint source bits",
        )
    widths = dict(zip(names, map(len, views), strict=True))
    shape = _layout(expr, widths)
    fraction = shape.fraction
    low, high = int(shape.low * (1 << fraction)), int(shape.high * (1 << fraction))
    width = max(1, high.bit_length(), (~low).bit_length() if low < 0 else 0) + (low < 0)
    if not 1 <= fraction < width <= 6:
        reject(
            compute.token,
            "precision",
            "inferred payoff requires up to six fixed-point bits",
        )
    evaluate = exact_expression(expr, set(names))
    actuals: list[Binding | None] = []
    for port, arg in zip(consumer.parameters, apply.arguments, strict=True):
        if _atom(arg, local.name.text):
            actuals.append(None)
        elif port.role == "real":
            actuals.append(owner.angle(arg, apply.token))
        elif port.role == "integer":
            actuals.append(owner.static_integer(arg))
        else:
            field = (
                owner.record_field(arg, allow_fractional=True)
                if arg.token.text == "."
                else None
            )
            view = field[1] if field else owner.array_view(arg)
            if field and field[0].kind not in {"qbit", "qbit[]", "qnum"}:
                reject(
                    arg.token,
                    "type",
                    "payoff needs a scalar bit or unsigned numeric view",
                )
            scale = field[0].fractional_bits if field else 0
            if set(view).intersection(selector):
                reject(
                    arg.token,
                    "alias",
                    "fixed-point payoff cannot write its source bits",
                )
            actuals.append(Word(len(view), scale, wires=view))
    payoff = _Payoff(owner, selector)
    table: dict[int, tuple[Operation, ...]] = {}
    for pattern in range(1 << len(selector)):
        env = {}
        offset = 0
        for name, count in widths.items():
            env[name] = Fraction((pattern >> offset) & ((1 << count) - 1))
            offset += count
        value = evaluate(env) * (1 << fraction)
        if value != int(value) or not 0 <= value < (1 << width):
            reject(
                compute.token,
                "precision",
                "payoff is outside its inferred nonnegative representation",
            )
        raw = int(value)
        if raw not in table:
            start, incoming = len(owner.operations), owner.effect
            try:
                virtual = Word(width, fraction, raw)
                payoff.call(
                    consumer, tuple(virtual if a is None else a for a in actuals), ()
                )
                table[raw] = tuple(owner.operations[start:])
            except (ArithmeticError, ValueError) as error:
                reject(
                    apply.token,
                    "angle",
                    f"invalid fixed-point payoff arithmetic: {error}",
                )
            finally:
                del owner.operations[start:]
                owner.effect = incoming
        _emit_branch(owner, apply.token, selector, pattern, table[raw])
    return True


def _emit_branch(
    owner: _Adapt,
    token: Token,
    selector: tuple[Location, ...],
    pattern: int,
    effects: tuple[Operation, ...],
) -> None:
    for effect in effects:
        if (
            effect.name == "qreg.control_x"
            and len(effect.attributes["indices"]) - 1 + len(selector) > 8
        ):
            indices = effect.attributes["indices"]
            wires = tuple(zip(effect.operands[1:], indices, strict=True))
            controls = (*selector, *wires[:-1])
            state = pattern | (effect.attributes["control_state"] << len(selector))
            negatives = tuple(
                location
                for bit, location in enumerate(controls)
                if not state & (1 << bit)
            )
            for value, wire in negatives:
                owner.effect_op("qreg.gate", token, (value,), gate="x", index=wire)
            owner.emit_indexed_x(token, (*controls, wires[-1]))
            for value, wire in reversed(negatives):
                owner.effect_op("qreg.gate", token, (value,), gate="x", index=wire)
        else:
            emit_selected(owner, token, selector, pattern, (effect,))
