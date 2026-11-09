"""Shared exact word inference and qint emission, independent of source syntax."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Set
from decimal import Decimal
from operator import and_, or_, xor
from types import MappingProxyType
from typing import Generic, NoReturn, TypeVar

from .._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)
from .profile import uint_type
from .ranges import bitwise_width, unsigned_width

_Node = TypeVar("_Node", bound=Hashable)
EXTREMA = frozenset({"min", "max"})

_WORD_OPERATORS = {"&": and_, "|": or_, "^": xor}

_RELATIONS = MappingProxyType(
    {
        "==": ("qint.equal", False, False),
        "!=": ("qint.equal", False, True),
        "<": ("qint.less", False, False),
        "<=": ("qint.less", True, True),
        ">": ("qint.less", True, False),
        ">=": ("qint.less", False, True),
    }
)


class WordExpression(Generic[_Node]):
    """Infer bounded arithmetic/predicate domains and emit existing qint records."""

    def __init__(
        self,
        node: _Node,
        widths: Mapping[str, int | None],
        name: str,
        signed_names: Set[str],
        *,
        text: Callable[[_Node], str],
        children: Callable[[_Node], tuple[_Node, ...]],
        location: Callable[[_Node], SourceLocation],
        reject: Callable[[_Node, str, str], NoReturn],
    ) -> None:
        self._text, self._children = text, children
        self._location, self._reject = location, reject
        self.node, self.widths, self.name = (node, widths, name)
        self.facts: dict[_Node, tuple[int, bool]] = {}
        self.fractional_bits: dict[_Node, int] = {}
        self.difference_ranges: dict[_Node, tuple[int, int]] = {}
        self.word_widths: dict[_Node, int] = {}
        self.signed_names = signed_names
        self.signed_words: set[_Node] = set()
        self.names: list[str] = []
        self.bound, self.quantum = self.infer(node)
        arguments: list[Value] = []
        for index, key in enumerate(self.names):
            width = widths[key]
            assert width is not None
            arguments.append(Value(ValueId(name, index), uint_type(width)))
        self.arguments = tuple(arguments)
        self.inputs = dict(zip(self.names, self.arguments, strict=True))
        self.operations: list[Operation] = []

    def infer(self, node: _Node) -> tuple[int, bool]:
        text = self._text(node)
        if text == "[":
            self._reject(node, "operator", "indexed values require a gate operand")
        if not self._children(node):
            if text.isdecimal():
                result = (int(text), False)
            elif "." in text:
                numerator, denominator = Decimal(text).as_integer_ratio()
                if denominator & denominator - 1:
                    self._reject(
                        node,
                        "precision",
                        "decimal operands must be exactly binary representable",
                    )
                self.fractional_bits[node] = denominator.bit_length() - 1
                result = (numerator, False)
            else:
                if text not in self.widths:
                    self._reject(node, "name", f"unknown output {text!r}")
                width = self.widths[text]
                if width is None:
                    self._reject(
                        node, "width", f"{text!r} has no initialized numeric width"
                    )
                if text not in self.names:
                    self.names.append(text)
                if text in self.signed_names:
                    self.signed_words.add(node)
                result = ((1 << width) - 1, True)
        elif text == "-":
            result = self.infer_difference(node)
        elif text == "~":
            operand = self._children(node)[0]
            value, quantum = self.infer(operand)
            if self.fractional_bits.get(operand) or (
                "." in self._text(operand) and (not self._children(operand))
            ):
                self._reject(
                    node, "precision", "bitwise complement requires an integer"
                )
            width = self.word_widths[operand]
            mask = (1 << width) - 1
            self.word_widths[node] = width
            if operand in self.signed_words:
                self.signed_words.add(node)
            result = (mask if quantum else mask ^ value, quantum)
        else:
            lhs, rhs = self._children(node)
            left, lq = self.infer(lhs)
            right, rq = self.infer(rhs)
            signed = (lhs in self.signed_words, rhs in self.signed_words)
            scale = max(
                self.fractional_bits.get(lhs, 0), self.fractional_bits.get(rhs, 0)
            )
            if (
                text not in EXTREMA
                and text != "*"
                and (
                    scale
                    or any(
                        "." in self._text(child) and (not self._children(child))
                        for child in self._children(node)
                    )
                )
            ):
                self._reject(
                    node,
                    "precision",
                    "fractional expressions are supported only in min/max, subtraction and unsigned multiplication",
                )
            if text in _WORD_OPERATORS:
                widths = (self.word_widths[lhs], self.word_widths[rhs])
                width = bitwise_width(text, widths, signed)
                signed_result = all(signed) if text == "&" else any(signed)
                if signed_result:
                    self.signed_words.add(node)
                self.word_widths[node] = width
                result = (
                    (
                        (1 << width) - 1
                        if lq or rq
                        else _WORD_OPERATORS[text](left, right)
                    ),
                    lq or rq,
                )
            elif text == "%":
                if (
                    any(signed)
                    or self._children(rhs)
                    or (not self._text(rhs).isdecimal())
                    or (right < 1)
                    or right & right - 1
                ):
                    self._reject(
                        node,
                        "modulo",
                        "modulo requires an unsigned dividend and a positive power-of-two literal divisor",
                    )
                mask = right - 1
                self.word_widths[node] = unsigned_width(mask)
                result = (mask if lq else left & mask, lq)
            elif text in {"==", "!="} and any(signed):
                width = self.equality_width(node)
                if width > 256:
                    self._reject(node, "width", "signed equality exceeds 256 bits")
                quantum = lq or rq
                result = (
                    (
                        1
                        if quantum
                        else int(
                            (self.integer_value(lhs) == self.integer_value(rhs))
                            == (text == "==")
                        )
                    ),
                    quantum,
                )
            elif any(signed):
                self._reject(
                    node,
                    "type",
                    "signed operands require bitwise operations, subtraction or negation",
                )
            elif text == "*":
                self.fractional_bits[node] = self.fractional_bits.get(
                    lhs, 0
                ) + self.fractional_bits.get(rhs, 0)
                result = (left * right, lq or rq)
            elif text in EXTREMA:
                self.fractional_bits[node] = scale
                left <<= scale - self.fractional_bits.get(lhs, 0)
                right <<= scale - self.fractional_bits.get(rhs, 0)
                if unsigned_width(max(left, right)) > 256:
                    self._reject(
                        node, "width", "scaled extremum operands exceed 256 bits"
                    )
                result = ((min if text == "min" else max)(left, right), lq or rq)
            elif text in _RELATIONS:
                opcode, swap, invert = _RELATIONS[text]
                a, b = (right, left) if swap else (left, right)
                bound = int(a == b if opcode == "qint.equal" else a < b) ^ int(invert)
                result = (1 if lq or rq else bound, lq or rq)
            else:
                bound = (
                    left + right
                    if text == "+"
                    else left ** int(self._text(self._children(node)[1]))
                )
                result = (bound, lq or rq)
        if unsigned_width(result[0]) > 256:
            self._reject(
                node, "width", "expression domain exceeds the 256-bit unsigned profile"
            )
        self.word_widths.setdefault(node, unsigned_width(result[0]))
        self.facts[node] = result
        return result

    def equality_width(self, node: _Node) -> int:
        widths = tuple(self.word_widths[child] for child in self._children(node))
        signed = tuple(child in self.signed_words for child in self._children(node))
        return bitwise_width("^", widths, signed)

    def integer_value(self, node: _Node) -> int:
        raw = self.facts[node][0]
        width = self.word_widths[node]
        if node in self.signed_words and raw >> width - 1:
            return raw - (1 << width)
        return raw

    def infer_difference(self, node: _Node) -> tuple[int, bool]:
        for child in self._children(node):
            self.infer(child)
        scale = max(
            self.fractional_bits.get(child, 0) for child in self._children(node)
        )
        self.fractional_bits[node] = scale
        intervals = []
        for child in self._children(node):
            bound, quantum = self.facts[child]
            if not quantum:
                low = high = self.integer_value(child)
            elif child in self.difference_ranges:
                low, high = self.difference_ranges[child]
            elif child in self.signed_words:
                sign_bit = 1 << self.word_widths[child] - 1
                low, high = (-sign_bit, sign_bit - 1)
            else:
                low, high = (0, bound)
            shift = scale - self.fractional_bits.get(child, 0)
            intervals.append((low << shift, high << shift))
        (a_min, a_max), (b_min, b_max) = (
            ((0, 0), intervals[0]) if len(intervals) == 1 else intervals
        )
        low, high = (a_min - b_max, a_max - b_min)
        self.difference_ranges[node] = (low, high)
        if low < 0:
            self.signed_words.add(node)
            width = max((~low).bit_length(), max(0, high).bit_length()) + 1
        else:
            width = unsigned_width(high)
        quantum = any(self.facts[child][1] for child in self._children(node))
        if width > 256:
            self._reject(node, "width", "difference exceeds the supported word width")
        self.word_widths[node] = width
        mask = (1 << width) - 1
        return (mask if quantum else low & mask, quantum)

    def operation(
        self,
        name: str,
        operands: tuple[Value, ...],
        kind: IRType,
        token: _Node,
        **attributes: object,
    ) -> Value:
        result = Value(
            ValueId(self.name, len(self.arguments) + len(self.operations)), kind
        )
        self.operations.append(
            Operation(
                name, operands, (result,), attributes, location=self._location(token)
            )
        )
        return result

    def emit_extremum(self, node: _Node, bound: int) -> Value:
        scale = self.fractional_bits[node]
        operands = []
        bounds = []
        for child in self._children(node):
            shift = scale - self.fractional_bits.get(child, 0)
            upper, quantum = self.facts[child]
            bounds.append(upper << shift)
            if not quantum:
                value = self.operation(
                    "arith.constant", (), INDEX, child, value=upper << shift
                )
            else:
                value = self.emit(child)
                if shift:
                    factor = self.operation(
                        "arith.constant", (), INDEX, node, value=1 << shift
                    )
                    value = self.operation(
                        "qint.multiply",
                        (value, factor),
                        uint_type(unsigned_width(upper << shift)),
                        node,
                    )
            operands.append(value)
        left, right = operands
        predicate = self.operation("qint.less", (left, right), uint_type(1), node)
        branches = (left, right) if self._text(node) == "min" else (right, left)
        width = unsigned_width(max(bounds))
        selected = self.operation(
            "qint.select", (predicate, *branches), uint_type(width), node
        )
        result_width = unsigned_width(bound)
        if result_width == width:
            return selected
        # Comparator selection proves min fits the smaller result bound.
        mask = self.operation(
            "arith.constant", (), INDEX, node, value=(1 << result_width) - 1
        )
        return self.operation(
            "qint.bitwise",
            (selected, mask),
            uint_type(result_width),
            node,
            operator="&",
        )

    def emit_difference(self, node: _Node) -> Value:
        width = self.word_widths[node]
        scale = self.fractional_bits[node]

        def constant(value: int) -> Value:
            return self.operation("arith.constant", (), INDEX, node, value=value)

        def mask_word(value: Value, bits: int, *, signed: bool = False) -> Value:
            return self.operation(
                "qint.bitwise",
                (value, constant((1 << bits) - 1)),
                uint_type(bits),
                node,
                operator="&",
                **{"signed": (True, False)} if signed else {},
            )

        def operand(child: _Node) -> Value:
            shift = scale - self.fractional_bits.get(child, 0)
            if not self.facts[child][1]:
                return constant(self.integer_value(child) << shift & (1 << width) - 1)
            value = self.emit(child)
            bits = width - shift
            signed = child in self.signed_words
            if bits <= 0:
                # Preserve the read even when every shifted bit is discarded.
                return self.operation(
                    "qint.bitwise",
                    (value, constant(0)),
                    uint_type(1),
                    node,
                    operator="&",
                )
            if (shift or signed) and self.word_widths[child] != bits:
                value = mask_word(value, bits, signed=signed)
            if shift:
                value = self.operation(
                    "qint.multiply",
                    (value, constant(1 << shift)),
                    uint_type(width),
                    node,
                )
            return value

        operands = tuple(operand(child) for child in self._children(node))
        if len(operands) == 1:
            operands = (constant(0), *operands)
        return self.operation("qint.subtract_mod", operands, uint_type(width), node)

    def emit_equality_operand(self, child: _Node, width: int) -> Value:
        if child not in self.signed_words or self.word_widths[child] == width:
            return self.emit(child)
        mask = (1 << width) - 1
        if not self.facts[child][1]:
            return self.operation(
                "arith.constant",
                (),
                INDEX,
                child,
                value=self.integer_value(child) & mask,
            )
        value = self.emit(child)
        extension = self.operation("arith.constant", (), INDEX, child, value=mask)
        return self.operation(
            "qint.bitwise",
            (value, extension),
            uint_type(width),
            child,
            operator="&",
            signed=(True, False),
        )

    def emit(self, node: _Node) -> Value:
        bound, quantum = self.facts[node]
        if not quantum:
            return self.operation("arith.constant", (), INDEX, node, value=bound)
        if not self._children(node):
            return self.inputs[self._text(node)]
        if self._text(node) in EXTREMA:
            return self.emit_extremum(node, bound)
        if self._text(node) == "-":
            return self.emit_difference(node)
        signed = tuple(child in self.signed_words for child in self._children(node))
        if self._text(node) in {"~", *_WORD_OPERATORS} and (
            any(signed)
            or any(self.word_widths[child] > 1 for child in self._children(node))
        ):
            return self.operation(
                "qint.bitwise",
                tuple(self.emit(child) for child in self._children(node)),
                uint_type(self.word_widths[node]),
                node,
                operator=self._text(node),
                **{"signed": signed} if any(signed) else {},
            )
        equality_width = (
            self.equality_width(node)
            if self._text(node) in {"==", "!="} and any(signed)
            else 0
        )
        left = (
            self.emit_equality_operand(self._children(node)[0], equality_width)
            if equality_width
            else self.emit(self._children(node)[0])
        )
        if self._text(node) == "%":
            divisor = self._children(node)[1]
            mask = self.operation(
                "arith.constant", (), INDEX, divisor, value=int(self._text(divisor)) - 1
            )
            return self.operation(
                "qint.bitwise",
                (left, mask),
                uint_type(self.word_widths[node]),
                node,
                operator="&",
            )
        if self._text(node) == "~":
            zero = self.operation("arith.constant", (), INDEX, node, value=0)
            return self.operation("qint.equal", (left, zero), uint_type(1), node)
        if self._text(node) == "**":
            result = left
            base_bound = self.facts[self._children(node)[0]][0]
            for exponent in range(2, int(self._text(self._children(node)[1])) + 1):
                result = self.operation(
                    "qint.multiply",
                    (result, left),
                    uint_type(unsigned_width(base_bound**exponent)),
                    node,
                )
            return result
        right = (
            self.emit_equality_operand(self._children(node)[1], equality_width)
            if equality_width
            else self.emit(self._children(node)[1])
        )
        if self._text(node) in {"&", "|", "^"}:
            if self._text(node) == "&":
                return self.operation(
                    "qint.multiply", (left, right), uint_type(1), node
                )
            zero = self.operation("arith.constant", (), INDEX, node, value=0)
            if self._text(node) == "|":
                total = self.operation("qint.add", (left, right), uint_type(2), node)
                return self.operation("qint.less", (zero, total), uint_type(1), node)
            equal = self.operation("qint.equal", (left, right), uint_type(1), node)
            return self.operation("qint.equal", (equal, zero), uint_type(1), node)
        if self._text(node) not in _RELATIONS:
            return self.operation(
                "qint.add" if self._text(node) == "+" else "qint.multiply",
                (left, right),
                uint_type(unsigned_width(bound)),
                node,
            )
        opcode, swap, invert = _RELATIONS[self._text(node)]
        result = self.operation(
            opcode, (right, left) if swap else (left, right), uint_type(1), node
        )
        if invert:
            zero = self.operation("arith.constant", (), INDEX, node, value=0)
            result = self.operation("qint.equal", (result, zero), uint_type(1), node)
        return result

    def function(self) -> HybridProgram:
        returned = self.emit(self.node)
        return HybridProgram(
            self.name,
            Region(
                (
                    Block(
                        self.arguments,
                        (
                            *self.operations,
                            Operation(
                                "qint.return",
                                (returned,),
                                location=self._location(self.node),
                            ),
                        ),
                    ),
                )
            ),
            location=self._location(self.node),
        )
