"""Native syntax to the admitted Compiler models, without synthesis or execution."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from operator import and_, or_, xor
from types import MappingProxyType

from ...compiler._construction.profile import uint_type
from ...compiler._construction.ranges import bitwise_width, unsigned_width
from ...compiler._construction.register_profile import (
    MAX_PREPARED_AMPLITUDES,
    equatorial_rotation,
    euler_rotation,
    register_type,
    register_width,
    verify_register_program,
)
from ...compiler._hybrid.model import (
    INDEX,
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    Value,
    ValueId,
)
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from .syntax import (
    ANGLE_CALLS,
    EXTREMA,
    Expression,
    Function,
    Parameter,
    Parser,
    Statement,
    Token,
    reject,
    tokenize,
)

SUPPORTED_SOURCE_VERSION = "2bfd835e73b6882fcc515a725098b04766c238e7"
MAX_EFFECT_OPERATIONS = 4096
MAX_QDRIFT_SAMPLES = 1024
QDRIFT_SEED = 0
_GATE_CALLS = frozenset({"X", "H", "S", "Z", "hadamard_transform"})
_PAIR_CALLS = frozenset({"CX", "SWAP", "CRX"})
_UNIFORM_CALLS = frozenset(
    {"prepare_uniform_interval_state", "prepare_uniform_trimmed_state"}
)
_INTRINSICS = (
    _UNIFORM_CALLS
    | _GATE_CALLS
    | _PAIR_CALLS
    | {
        "allocate",
        "encode_in_angle",
        "encode_on_bloch",
        "phase",
        "pi",
        "control",
        "prepare_ghz_state",
        "prepare_amplitudes",
        "prepare_state",
        "prepare_bell_state",
        "suzuki_trotter",
        "single_pauli_exponent",
        "exponentiate",
        "qdrift",
        "qft",
        "qct_qst_type1",
        "qct_type2",
        "qst_type2",
        "qct_qst_type2",
    }
    | EXTREMA
    | ANGLE_CALLS
)


def _literal_angle(angle: Expression, gate: Token) -> float:
    if (
        angle.children
        or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", angle.token.text)
        or Decimal(angle.token.text) > 2**53
    ):
        reject(
            angle.token,
            "angle",
            f"{gate.text} angle must be a literal in 0..2**53 radians",
        )
    return float(angle.token.text)


def _pauli_support(
    axes: Expression, width: int, *, sparse: bool
) -> tuple[tuple[int, str], ...]:
    if not sparse:
        if len(axes.children) != width:
            reject(axes.token, "pauli", "Pauli word length must match the array width")
        return tuple(
            (index, axis.token.text)
            for index, axis in enumerate(axes.children)
            if axis.token.text != "I"
        )
    seen = set()
    support = []
    for indexed in axes.children:
        axis, index_arg = indexed.children
        index_token = index_arg.token
        if index_arg.children or not index_token.text.isdecimal():
            reject(
                index_token,
                "pauli",
                "Pauli index must be a nonnegative integer literal",
            )
        index = int(index_token.text)
        if index >= width or index in seen:
            reject(index_token, "pauli", "Pauli indices must be distinct and in bounds")
        seen.add(index)
        if axis.token.text != "I":
            support.append((index, axis.token.text))
    return tuple(support)


def _sparse_width(node: Expression) -> int:
    if node.children or not node.token.text.isdecimal():
        reject(node.token, "pauli", "num_qubits must be an integer literal in 1..256")
    width = int(node.token.text)
    if not 1 <= width <= 256:
        reject(node.token, "pauli", "num_qubits must be an integer literal in 1..256")
    return width


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


@dataclass(frozen=True)
class QmodImport:
    """Private construction handoff; source_version is declared, not authenticated."""

    program: HybridProgram
    functions: tuple[HybridProgram, ...]
    source_version: str
    inferred_widths: Mapping[str, int]
    source_name: str
    fractional_bits: Mapping[str, int]
    signed_registers: tuple[str, ...]


class _Expression:
    """Infer bounded arithmetic/predicate domains and emit existing qint records."""

    def __init__(
        self,
        node: Expression,
        widths: Mapping[str, int | None],
        name: str,
        signed_names: set[str],
    ) -> None:
        self.node, self.widths, self.name = node, widths, name
        self.facts: dict[Expression, tuple[int, bool]] = {}
        self.fractional_bits: dict[Expression, int] = {}
        self.difference_ranges: dict[Expression, tuple[int, int]] = {}
        self.word_widths: dict[Expression, int] = {}
        self.signed_names = signed_names
        self.signed_words: set[Expression] = set()
        self.names: list[str] = []
        self.bound, self.quantum = self.infer(node)
        arguments: list[Value] = []
        for index, key in enumerate(self.names):
            width = widths[key]
            assert width is not None  # infer already refuses unresolved inputs.
            arguments.append(Value(ValueId(name, index), uint_type(width)))
        self.arguments = tuple(arguments)
        self.inputs = dict(zip(self.names, self.arguments, strict=True))
        self.operations: list[Operation] = []

    def infer(self, node: Expression) -> tuple[int, bool]:
        text = node.token.text
        if text == "[":
            reject(node.token, "operator", "indexed values require a gate operand")
        if not node.children:
            if text.isdecimal():
                result = (int(text), False)
            elif "." in text:
                numerator, denominator = Decimal(text).as_integer_ratio()
                if denominator & (denominator - 1):
                    reject(
                        node.token,
                        "precision",
                        "decimal operands must be exactly binary representable",
                    )
                self.fractional_bits[node] = denominator.bit_length() - 1
                result = (numerator, False)
            else:
                if text not in self.widths:
                    reject(node.token, "name", f"unknown output {text!r}")
                width = self.widths[text]
                if width is None:
                    reject(
                        node.token,
                        "width",
                        f"{text!r} has no initialized numeric width",
                    )
                if text not in self.names:
                    self.names.append(text)
                if text in self.signed_names:
                    self.signed_words.add(node)
                result = ((1 << width) - 1, True)
        elif text == "-":
            result = self.infer_difference(node)
        elif text == "~":
            operand = node.children[0]
            value, quantum = self.infer(operand)
            if self.fractional_bits.get(operand) or (
                "." in operand.token.text and not operand.children
            ):
                reject(
                    node.token, "precision", "bitwise complement requires an integer"
                )
            width = self.word_widths[operand]
            mask = (1 << width) - 1
            self.word_widths[node] = width
            if operand in self.signed_words:
                self.signed_words.add(node)
            result = (mask if quantum else mask ^ value, quantum)
        else:
            lhs, rhs = node.children
            left, lq = self.infer(lhs)
            right, rq = self.infer(rhs)
            signed = lhs in self.signed_words, rhs in self.signed_words
            scale = max(
                self.fractional_bits.get(lhs, 0), self.fractional_bits.get(rhs, 0)
            )
            if (
                text not in EXTREMA
                and text != "*"
                and (
                    scale
                    or any(
                        "." in child.token.text and not child.children
                        for child in node.children
                    )
                )
            ):
                reject(
                    node.token,
                    "precision",
                    "fractional expressions are supported only in min/max, subtraction and unsigned multiplication",
                )
            if text in _WORD_OPERATORS:
                widths = self.word_widths[lhs], self.word_widths[rhs]
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
                    or rhs.children
                    or not rhs.token.text.isdecimal()
                    or right < 1
                    or right & (right - 1)
                ):
                    reject(
                        node.token,
                        "modulo",
                        "modulo requires an unsigned dividend and a positive power-of-two literal divisor",
                    )
                mask = right - 1
                self.word_widths[node] = unsigned_width(mask)
                result = (mask if lq else left & mask, lq)
            elif text in {"==", "!="} and any(signed):
                width = self.equality_width(node)
                if width > 256:
                    reject(node.token, "width", "signed equality exceeds 256 bits")
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
                reject(
                    node.token,
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
                    reject(
                        node.token, "width", "scaled extremum operands exceed 256 bits"
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
                    else left ** int(node.children[1].token.text)
                )
                result = (bound, lq or rq)
        if unsigned_width(result[0]) > 256:
            reject(
                node.token,
                "width",
                "expression domain exceeds the 256-bit unsigned profile",
            )
        self.word_widths.setdefault(node, unsigned_width(result[0]))
        self.facts[node] = result
        return result

    def equality_width(self, node: Expression) -> int:
        widths = tuple(self.word_widths[child] for child in node.children)
        signed = tuple(child in self.signed_words for child in node.children)
        return bitwise_width("^", widths, signed)

    def integer_value(self, node: Expression) -> int:
        raw = self.facts[node][0]
        width = self.word_widths[node]
        if node in self.signed_words and raw >> (width - 1):
            return raw - (1 << width)
        return raw

    def infer_difference(self, node: Expression) -> tuple[int, bool]:
        for child in node.children:
            self.infer(child)
        scale = max(self.fractional_bits.get(child, 0) for child in node.children)
        self.fractional_bits[node] = scale
        intervals = []
        for child in node.children:
            bound, quantum = self.facts[child]
            if not quantum:
                low = high = self.integer_value(child)
            elif child in self.difference_ranges:
                low, high = self.difference_ranges[child]
            elif child in self.signed_words:
                sign_bit = 1 << (self.word_widths[child] - 1)
                low, high = -sign_bit, sign_bit - 1
            else:
                low, high = 0, bound
            shift = scale - self.fractional_bits.get(child, 0)
            intervals.append((low << shift, high << shift))
        (a_min, a_max), (b_min, b_max) = (
            ((0, 0), intervals[0]) if len(intervals) == 1 else intervals
        )
        low, high = a_min - b_max, a_max - b_min
        self.difference_ranges[node] = low, high
        if low < 0:
            self.signed_words.add(node)
            width = max((~low).bit_length(), max(0, high).bit_length()) + 1
        else:
            width = unsigned_width(high)
        quantum = any(self.facts[child][1] for child in node.children)
        if width > 256:
            reject(node.token, "width", "difference exceeds the supported word width")
        self.word_widths[node] = width
        mask = (1 << width) - 1
        return (mask if quantum else low & mask), quantum

    def operation(
        self,
        name: str,
        operands: tuple[Value, ...],
        kind: IRType,
        token: Token,
        **attributes: object,
    ) -> Value:
        result = Value(
            ValueId(self.name, len(self.arguments) + len(self.operations)), kind
        )
        self.operations.append(
            Operation(name, operands, (result,), attributes, location=token.location)
        )
        return result

    def emit_extremum(self, node: Expression, bound: int) -> Value:
        scale = self.fractional_bits[node]
        operands = []
        bounds = []
        for child in node.children:
            shift = scale - self.fractional_bits.get(child, 0)
            upper, quantum = self.facts[child]
            bounds.append(upper << shift)
            if not quantum:
                value = self.operation(
                    "arith.constant", (), INDEX, child.token, value=upper << shift
                )
            else:
                value = self.emit(child)
                if shift:
                    factor = self.operation(
                        "arith.constant", (), INDEX, node.token, value=1 << shift
                    )
                    value = self.operation(
                        "qint.multiply",
                        (value, factor),
                        uint_type(unsigned_width(upper << shift)),
                        node.token,
                    )
            operands.append(value)
        left, right = operands
        predicate = self.operation("qint.less", (left, right), uint_type(1), node.token)
        branches = (left, right) if node.token.text == "min" else (right, left)
        width = unsigned_width(max(bounds))
        selected = self.operation(
            "qint.select", (predicate, *branches), uint_type(width), node.token
        )
        result_width = unsigned_width(bound)
        if result_width == width:
            return selected
        # The generic selector sees either branch's full domain. For min, the
        # comparator proves that the selected value fits the smaller bound.
        mask = self.operation(
            "arith.constant", (), INDEX, node.token, value=(1 << result_width) - 1
        )
        return self.operation(
            "qint.bitwise",
            (selected, mask),
            uint_type(result_width),
            node.token,
            operator="&",
        )

    def emit_difference(self, node: Expression) -> Value:
        width = self.word_widths[node]
        scale = self.fractional_bits[node]

        def constant(value: int) -> Value:
            return self.operation("arith.constant", (), INDEX, node.token, value=value)

        def mask_word(value: Value, bits: int, *, signed: bool = False) -> Value:
            return self.operation(
                "qint.bitwise",
                (value, constant((1 << bits) - 1)),
                uint_type(bits),
                node.token,
                operator="&",
                **({"signed": (True, False)} if signed else {}),
            )

        def operand(child: Expression) -> Value:
            shift = scale - self.fractional_bits.get(child, 0)
            if not self.facts[child][1]:
                return constant(
                    (self.integer_value(child) << shift) & ((1 << width) - 1)
                )
            value = self.emit(child)
            bits = width - shift
            signed = child in self.signed_words
            if bits <= 0:
                # Keep the read in the graph even when every shifted bit is discarded.
                return self.operation(
                    "qint.bitwise",
                    (value, constant(0)),
                    uint_type(1),
                    node.token,
                    operator="&",
                )
            if (shift or signed) and self.word_widths[child] != bits:
                value = mask_word(value, bits, signed=signed)
            if shift:
                value = self.operation(
                    "qint.multiply",
                    (value, constant(1 << shift)),
                    uint_type(width),
                    node.token,
                )
            return value

        operands = tuple(operand(child) for child in node.children)
        if len(operands) == 1:
            operands = (constant(0), *operands)
        return self.operation(
            "qint.subtract_mod", operands, uint_type(width), node.token
        )

    def emit_equality_operand(self, child: Expression, width: int) -> Value:
        if child not in self.signed_words or self.word_widths[child] == width:
            return self.emit(child)
        mask = (1 << width) - 1
        if not self.facts[child][1]:
            return self.operation(
                "arith.constant",
                (),
                INDEX,
                child.token,
                value=self.integer_value(child) & mask,
            )
        value = self.emit(child)
        extension = self.operation("arith.constant", (), INDEX, child.token, value=mask)
        return self.operation(
            "qint.bitwise",
            (value, extension),
            uint_type(width),
            child.token,
            operator="&",
            signed=(True, False),
        )

    def emit(self, node: Expression) -> Value:
        bound, quantum = self.facts[node]
        if not quantum:
            return self.operation("arith.constant", (), INDEX, node.token, value=bound)
        if not node.children:
            return self.inputs[node.token.text]
        if node.token.text in EXTREMA:
            return self.emit_extremum(node, bound)
        if node.token.text == "-":
            return self.emit_difference(node)
        signed = tuple(child in self.signed_words for child in node.children)
        if node.token.text in {"~", *_WORD_OPERATORS} and (
            any(signed) or any(self.word_widths[child] > 1 for child in node.children)
        ):
            return self.operation(
                "qint.bitwise",
                tuple(self.emit(child) for child in node.children),
                uint_type(self.word_widths[node]),
                node.token,
                operator=node.token.text,
                **({"signed": signed} if any(signed) else {}),
            )
        equality_width = (
            self.equality_width(node)
            if node.token.text in {"==", "!="} and any(signed)
            else 0
        )
        left = (
            self.emit_equality_operand(node.children[0], equality_width)
            if equality_width
            else self.emit(node.children[0])
        )
        if node.token.text == "%":
            divisor = node.children[1]
            mask = self.operation(
                "arith.constant",
                (),
                INDEX,
                divisor.token,
                value=int(divisor.token.text) - 1,
            )
            return self.operation(
                "qint.bitwise",
                (left, mask),
                uint_type(self.word_widths[node]),
                node.token,
                operator="&",
            )
        if node.token.text == "~":
            zero = self.operation("arith.constant", (), INDEX, node.token, value=0)
            return self.operation("qint.equal", (left, zero), uint_type(1), node.token)
        if node.token.text == "**":
            result = left
            base_bound = self.facts[node.children[0]][0]
            for exponent in range(2, int(node.children[1].token.text) + 1):
                result = self.operation(
                    "qint.multiply",
                    (result, left),
                    uint_type(unsigned_width(base_bound**exponent)),
                    node.token,
                )
            return result
        right = (
            self.emit_equality_operand(node.children[1], equality_width)
            if equality_width
            else self.emit(node.children[1])
        )
        if node.token.text in {"&", "|", "^"}:
            if node.token.text == "&":
                return self.operation(
                    "qint.multiply", (left, right), uint_type(1), node.token
                )
            zero = self.operation("arith.constant", (), INDEX, node.token, value=0)
            if node.token.text == "|":
                total = self.operation(
                    "qint.add", (left, right), uint_type(2), node.token
                )
                return self.operation(
                    "qint.less", (zero, total), uint_type(1), node.token
                )
            equal = self.operation(
                "qint.equal", (left, right), uint_type(1), node.token
            )
            return self.operation("qint.equal", (equal, zero), uint_type(1), node.token)
        if node.token.text not in _RELATIONS:
            return self.operation(
                "qint.add" if node.token.text == "+" else "qint.multiply",
                (left, right),
                uint_type(unsigned_width(bound)),
                node.token,
            )
        opcode, swap, invert = _RELATIONS[node.token.text]
        result = self.operation(
            opcode, (right, left) if swap else (left, right), uint_type(1), node.token
        )
        if invert:
            zero = self.operation("arith.constant", (), INDEX, node.token, value=0)
            result = self.operation(
                "qint.equal", (result, zero), uint_type(1), node.token
            )
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
                                location=self.node.token.location,
                            ),
                        ),
                    ),
                )
            ),
            location=self.node.token.location,
        )


class _Adapt:
    def __init__(
        self,
        syntax: Function,
        resolve: Callable[[Token], tuple[Function, HybridProgram]],
        definitions: Mapping[str, Function],
        constants: Mapping[str, Expression],
    ) -> None:
        self.syntax = syntax
        self.resolve = resolve
        self.definitions = definitions
        self.constants = constants
        self.classical_values: dict[str, float] = {}
        self.classical_arrays: dict[str, tuple[float, ...]] = {}
        self.scope = syntax.name.text
        self.parameters: dict[str, Parameter] = {}
        for parameter in syntax.parameters:
            name = parameter.name.text
            if name in self.parameters:
                reject(parameter.name, "name", f"duplicate output {name!r}")
            self.parameters[name] = parameter
        self.widths = {name: p.width for name, p in self.parameters.items()}
        self.inferred: dict[str, int] = {}
        self.signed: set[str] = set()
        self.fractional_bits: dict[str, int] = {}
        self.registers: dict[str, Value] = {}
        self.counter = len(self.parameters) + 1
        self.entry = Value(ValueId(self.scope, 0), QUANTUM_EFFECT)
        self.effect = self.entry
        self.operations: list[Operation] = []
        self.functions: list[HybridProgram] = []
        self.loop_indices: dict[str, int] = {}
        self.local_scope: list[tuple[str, Token]] | None = None

    def register(
        self,
        token: Token,
        inferred: int | None = None,
        *,
        allow_qubit_array: bool = False,
        allow_fractional_output: bool = False,
    ) -> Value:
        name = token.text
        if name in self.fractional_bits and not allow_fractional_output:
            reject(token, "precision", "a fractional result is a terminal output")
        if name not in self.parameters:
            reject(token, "name", f"unknown output {name!r}")
        if self.parameters[name].kind == "qbit[]" and not allow_qubit_array:
            reject(
                token,
                "type",
                "qubit arrays support allocation and whole-array Hadamard",
            )
        if self.parameters[name].role == "real":
            reject(token, "type", "classical real is not a quantum register")
        local = self.parameters[name].role == "local"
        if local and self.local_scope is None:
            reject(
                token,
                "scope",
                "local references are available only inside within/apply",
            )
        width = self.widths[name]
        if width is None and inferred is not None:
            self.widths[name] = width = inferred
            if not local:
                self.inferred[name] = inferred
        if width is None:
            reject(
                token,
                "width",
                f"{name!r} needs a declared width or explicit initialization size",
            )
        if inferred is not None and inferred > width:
            reject(
                token,
                "width",
                f"declared width {width} cannot fit the expression domain requiring {inferred} bits",
            )
        if name not in self.registers:
            if local:
                index = self.counter
                self.counter += 1
            else:
                index = tuple(self.parameters).index(name) + 1
            self.registers[name] = Value(
                ValueId(self.scope, index), register_type(width)
            )
            if local:
                assert self.local_scope is not None
                self.local_scope.append((name, token))
                self.append_op(
                    token,
                    Operation(
                        "qreg.local",
                        results=(self.registers[name],),
                        attributes={"name": f"{name}.within{index}"},
                        location=token.location,
                    ),
                )
        return self.registers[name]

    def append_op(self, token: Token, operation: Operation) -> None:
        if len(self.operations) >= MAX_EFFECT_OPERATIONS:
            reject(
                token,
                "budget.expansion",
                "native expansion exceeds 4096 effect operations",
            )
        self.operations.append(operation)

    def effect_op(
        self, name: str, token: Token, values: tuple[Value, ...], **attributes: object
    ) -> None:
        effect = Value(ValueId(self.scope, self.counter), QUANTUM_EFFECT)
        self.counter += 1
        self.append_op(
            token,
            Operation(
                name,
                (self.effect, *values),
                (effect,),
                attributes,
                location=token.location,
            ),
        )
        self.effect = effect

    @staticmethod
    def atom(node: Expression) -> Token:
        if node.children or node.token.text.isdecimal():
            reject(node.token, "signature", "this argument must be one output variable")
        return node.token

    def static_integer(self, node: Expression) -> int:
        text = node.token.text
        if text.isdecimal():
            return int(text)
        if text in {"+", "-"}:
            left, right = (self.static_integer(child) for child in node.children)
            return left + right if text == "+" else left - right
        if text == ".":
            name = node.children[0].token.text
            width = self.widths.get(name)
            if width is None or self.parameters[name].kind != "qbit[]":
                reject(
                    node.token, "width", "len requires a qubit array with a known width"
                )
            return width
        if text in self.loop_indices:
            return self.loop_indices[text]
        reject(node.token, "expression", "expected a static integer or loop index")

    def gate_target(self, node: Expression) -> tuple[Value, int]:
        if node.token.text == "[":
            array, offset = node.children
            target = self.atom(array)
            value = self.register(target, allow_qubit_array=True)
            if self.parameters[target.text].kind != "qbit[]":
                reject(target, "operator", "indexing requires a qubit array")
            index = self.static_integer(offset)
            width = self.widths[target.text]
            assert width is not None
            if not 0 <= index < width:
                reject(offset.token, "width", "qubit array index is out of bounds")
            return value, index
        target = self.atom(node)
        value = self.register(target)
        if self.parameters[target.text].kind != "qbit":
            reject(
                target, "type", "gate operand requires a scalar qubit or array element"
            )
        return value, 0

    def repeat(self, statement: Statement, index: str) -> None:
        variable = statement.repeat_index
        assert variable is not None
        name = variable.text
        if name in self.parameters or name in _INTRINSICS or name in self.constants:
            reject(variable, "name", "loop index cannot shadow a register or intrinsic")
        count = self.static_integer(statement.arguments[0])
        if not 0 <= count <= 256:
            reject(
                statement.token, "budget.expansion", "repeat count must be in 0..256"
            )
        for item in statement.repeat_body:
            if (
                item.declaration is not None
                or item.token.text not in _GATE_CALLS | _PAIR_CALLS | ANGLE_CALLS
            ):
                reject(item.token, "scope", "repeat bodies admit only gates")
        for iteration in range(count):
            self.loop_indices[name] = iteration
            for ordinal, item in enumerate(statement.repeat_body):
                self.statement(item, f"{index}.{iteration}.{ordinal}")
        self.loop_indices.pop(name, None)

    def angle(self, node: Expression, gate: Token) -> float:
        if not node.children and node.token.text in self.classical_values:
            return self.classical_values[node.token.text]
        if node.token.text == "[":
            array, index = node.children
            values = self.classical_arrays.get(array.token.text)
            if values is not None and not array.children:
                if index.children or not index.token.text.isdecimal():
                    reject(
                        index.token,
                        "binding",
                        "real array index requires an integer literal",
                    )
                offset = int(index.token.text)
                if offset >= len(values):
                    reject(index.token, "binding", "real array index is out of bounds")
                return values[offset]
        return _literal_angle(node, gate)

    def encode_data(self, token: Token, args: tuple[Expression, ...]) -> None:
        if len(args) != 2:
            reject(token, "signature", "data encoding requires a real array and output")
        source, output = (self.atom(arg) for arg in args)
        values = self.classical_arrays.get(source.text)
        if values is None:
            reject(source, "binding", "encoding requires a bound real array")
        if (
            output.text not in self.parameters
            or self.parameters[output.text].kind != "qbit[]"
        ):
            reject(output, "type", "encoding requires a qubit array output")
        bloch = token.text == "encode_on_bloch"
        width = (len(values) + 1) // 2 if bloch else len(values)
        target = self.register(output, width, allow_qubit_array=True)
        if register_width(target) != width:
            reject(output, "width", "encoding output width must match its data")
        self.effect_op("qreg.allocate", token, (target,))
        for index, value in enumerate(values):
            if bloch:
                gate = "rz" if index % 2 else "rx"
                scale = (
                    math.tau
                    if index == len(values) - 1 and len(values) % 2
                    else math.pi
                )
                wire = index // 2
            else:
                gate, scale, wire = "ry", math.pi, index
            self.effect_op(
                "qreg.gate",
                token,
                (target,),
                gate=gate,
                index=wire,
                angle=scale * value,
            )

    def call_bound_helper(
        self, syntax: Function, token: Token, arguments: tuple[Expression, ...]
    ) -> None:
        if len(arguments) != len(syntax.parameters):
            reject(token, "signature", "helper argument count disagrees")
        child = _Adapt(syntax, self.resolve, self.definitions, self.constants)
        for argument, parameter in zip(arguments, syntax.parameters, strict=True):
            name = parameter.name.text
            if parameter.role == "real":
                child.classical_values[name] = self.angle(argument, token)
            else:
                target = self.atom(argument)
                value = self.register(target, allow_qubit_array=True)
                if self.parameters[target.text].kind != "qbit[]":
                    reject(target, "type", "helper requires a qubit array")
                child.registers[name] = value
                child.widths[name] = self.widths[target.text]
        for index, statement in enumerate(syntax.statements):
            child.statement(statement, str(index))
        for op in child.operations:
            # Replay the specialized gates with the caller's linear effect.
            self.effect_op(op.name, token, op.operands[1:], **op.attributes)

    def call(
        self,
        token: Token,
        arguments: tuple[Expression, ...],
        *,
        const_only: bool = False,
        control_bit: Value | None = None,
        condition_inputs: tuple[Value, ...] = (),
    ) -> None:
        if token.text in self.parameters:
            reject(token, "call", "an output variable shadows this helper call")
        definition = self.definitions.get(token.text)
        if (
            definition is not None
            and token.text != "main"
            and _bound_helper(definition)
        ):
            if const_only or control_bit is not None:
                reject(
                    token,
                    "scope",
                    "classical and writable-array helpers require ordinary calls",
                )
            self.call_bound_helper(definition, token, arguments)
            return
        syntax, program = self.resolve(token)
        if control_bit is not None and any(
            p.role != "const" for p in syntax.parameters
        ):
            reject(token, "control", "controlled helpers require Const-only ports")
        if const_only and any(p.role != "const" for p in syntax.parameters):
            reject(token, "scope", "apply helper calls require Const-only parameters")
        if len(arguments) != len(syntax.parameters):
            reject(token, "signature", "helper argument count disagrees")
        const = iter(program.body.blocks[0].arguments[1:])
        outputs = iter(
            op.results[0]
            for op in program.body.blocks[0].operations
            if op.name == "qreg.declare"
        )
        reads: list[Value] = []
        writes: list[Value] = []
        for argument, parameter in zip(arguments, syntax.parameters, strict=True):
            borrowed = parameter.role == "const"
            formal = next(const if borrowed else outputs)
            actual_name = self.atom(argument)
            actual = self.register(
                actual_name, None if borrowed else formal.type.parameters[0]
            )
            if (
                actual_name.text in self.signed
                or self.parameters[actual_name.text].kind != parameter.kind
                or actual.type != formal.type
            ):
                reject(
                    argument.token,
                    "signature",
                    "helper argument kind/width must match exactly",
                )
            (reads if borrowed else writes).append(actual)
        if control_bit is None:
            self.effect_op("qreg.call", token, (*reads, *writes), callee=program.name)
        else:
            if len(set(reads)) != len(reads) or set(reads).intersection(
                condition_inputs
            ):
                reject(token, "control", "controlled helper arguments must be disjoint")
            self.effect_op(
                "qreg.control_call", token, (control_bit, *reads), callee=program.name
            )

    def scalar_control(self, statement: Statement, index: str) -> None:
        token = statement.token
        body = statement.control_body
        if len(statement.arguments) != 1 or body is None or len(body) != 1:
            reject(token, "control", "control requires one condition and one operation")
        condition = statement.arguments[0]
        parameter = self.parameters.get(condition.token.text)
        if (
            not condition.children
            and parameter is not None
            and parameter.kind == "qbit[]"
        ):
            operation = body[0]
            if operation.token.text != "X" or len(operation.arguments) != 1:
                reject(
                    token, "control", "array control requires a single scalar X target"
                )
            target = self.atom(operation.arguments[0])
            if target.text == parameter.name.text:
                reject(token, "control", "condition and target must be disjoint")
            control = self.register(parameter.name, allow_qubit_array=True)
            destination, _ = self.gate_target(operation.arguments[0])
            self.effect_op("qreg.control_x", token, (control, destination))
            return
        expression = _Expression(
            condition, self.widths, f"{self.scope}.control{index}", self.signed
        )
        if (
            condition in expression.signed_words
            or expression.fractional_bits.get(condition, 0)
            or not expression.quantum
            or unsigned_width(expression.bound) != 1
        ):
            reject(token, "control", "control requires a quantum one-bit condition")
        inputs = tuple(
            self.register(self.parameters[name].name) for name in expression.names
        )
        if sum(value.type.parameters[0] for value in inputs) > 8:
            reject(token, "control", "control predicate exceeds eight input bits")
        operation = body[0]
        args = operation.arguments
        targets: tuple[Value, ...] = ()
        angle = None
        if operation.token.text in {"H", "hadamard_transform"} and len(args) == 1:
            targets = (self.register(self.atom(args[0])),)
            opcode = "qreg.control_h"
        elif operation.token.text == "phase" and len(args) == 2:
            coefficient = args[1]
            if (
                not coefficient.children
                and coefficient.token.text == "pi"
                and "pi" not in self.parameters
            ):
                angle = math.pi
            else:
                angle = _literal_angle(coefficient, operation.token)
            value = args[0]
            if not value.children and value.token.text in {"0", "1"}:
                angle *= int(value.token.text)
            else:
                targets = (self.register(self.atom(value)),)
            opcode = "qreg.control_phase"
        elif operation.token.text not in _INTRINSICS and operation.token.text not in {
            "measure",
            "release",
            "consume",
        }:
            opcode = "qreg.control_call"
        else:
            reject(token, "control", "expected a scalar Hadamard or phase")
        if any(target in inputs for target in targets):
            reject(token, "control", "condition and target must be disjoint")
        computed = bool(condition.children)
        if computed:
            graph = expression.function()
            self.functions.append(graph)
            predicate = Value(ValueId(self.scope, self.counter), register_type(1))
            self.counter += 1
            self.append_op(
                token,
                Operation(
                    "qreg.local",
                    results=(predicate,),
                    attributes={"name": f"control.{predicate.id.index}"},
                    location=token.location,
                ),
            )
            self.effect_op(
                "qreg.compute", token, (predicate, *inputs), callee=graph.name
            )
        else:
            predicate = inputs[0]
        phase_attributes = {} if angle is None else {"angle": angle}
        if opcode == "qreg.control_call":
            self.call(
                operation.token, args, control_bit=predicate, condition_inputs=inputs
            )
        else:
            self.effect_op(opcode, token, (predicate, *targets), **phase_attributes)
        if computed:
            self.effect_op("qreg.uncompute", token, (predicate,))
            self.effect_op("qreg.release", token, (predicate,))

    def within(self, statement: Statement, index: str) -> None:
        if not statement.within:
            reject(
                statement.token,
                "scope",
                "within requires at least one local computation",
            )
        parent_scope = self.local_scope
        self.local_scope = []
        try:
            for position, child in enumerate(statement.within):
                if child.token.text != "=":
                    reject(
                        child.token,
                        "scope",
                        "within admits only quantum-expression local assignments",
                    )
                target = self.atom(child.arguments[0])
                if (
                    target.text not in self.parameters
                    or self.parameters[target.text].role != "local"
                ):
                    reject(target, "scope", "within prepares declared locals only")
                self.statement(child, f"{index}.within{position}")
            for position, child in enumerate(statement.apply):
                if child.within is not None:
                    self.within(child, f"{index}.apply{position}")
                elif child.token.text == "phase":
                    self.statement(child, f"{index}.apply{position}")
                elif child.token.text in _INTRINSICS or child.token.text == "=":
                    reject(
                        child.token,
                        "scope",
                        "apply admits phase and Const-only helper calls",
                    )
                else:
                    self.call(child.token, child.arguments, const_only=True)
            for name, token in reversed(self.local_scope):
                value = self.registers[name]
                self.effect_op("qreg.uncompute", token, (value,))
                self.effect_op("qreg.release", token, (value,))
        finally:
            for name, _ in self.local_scope:
                self.registers.pop(name, None)
            self.local_scope = parent_scope

    def evolution_angle(self, node: Expression, token: Token) -> float:
        if node.token.text == "-":
            return -self.angle(node.children[0], token)
        return self.angle(node, token)

    def single_pauli_exponent(self, token: Token, args: tuple[Expression, ...]) -> None:
        if len(args) != 3:
            reject(
                token,
                "signature",
                "single_pauli_exponent requires a Pauli word, coefficient and target",
            )
        axes, coefficient, target_arg = args
        target = self.atom(target_arg)
        register = self.register(target, allow_qubit_array=True)
        if self.parameters[target.text].kind != "qbit[]":
            reject(target, "type", "single_pauli_exponent requires a qubit array")
        support = _pauli_support(axes, register_width(register), sparse=False)
        angle = self.evolution_angle(coefficient, token)
        if support and not math.isfinite(2 * angle):
            reject(
                coefficient.token, "angle", "Pauli rotation angle exceeds finite range"
            )
        self.pauli_exponential(token, register, support, angle)

    def suzuki_trotter(self, token: Token, args: tuple[Expression, ...]) -> None:
        if len(args) != 5:
            reject(
                token,
                "signature",
                "suzuki_trotter requires terms, time, order, repetitions and target",
            )
        terms, time_arg, order_arg, repetitions_arg, target_arg = args
        if any(
            arg.children or not arg.token.text.isdecimal()
            for arg in (order_arg, repetitions_arg)
        ):
            reject(token, "evolution", "order and repetitions must be integer literals")
        order = int(order_arg.token.text)
        repetitions = int(repetitions_arg.token.text)
        if order not in {1, 2} or not 1 <= repetitions <= 256:
            reject(token, "evolution", "use order 1 or 2 and 1..256 repetitions")
        self.pauli_evolution(token, terms, time_arg, target_arg, order, repetitions)

    def exponentiate(self, token: Token, args: tuple[Expression, ...]) -> None:
        if len(args) != 3:
            reject(
                token, "signature", "exponentiate requires an operator, time and target"
            )
        operator, time_arg, target_arg = args
        # Classiq's Hamiltonian Simulation guide specifies a first-order single step.
        self.pauli_evolution(token, operator, time_arg, target_arg, 1, 1)

    def pauli_evolution(
        self,
        token: Token,
        operator: Expression,
        time_arg: Expression,
        target_arg: Expression,
        order: int,
        repetitions: int,
    ) -> None:
        register, terms = self.pauli_terms(token, operator, target_arg)
        step = self.evolution_angle(time_arg, token) / repetitions
        if order == 2:
            step /= 2
        sequence = terms if order == 1 else [*terms, *reversed(terms)]
        for _ in range(repetitions):
            for support, weight in sequence:
                self.pauli_exponential(token, register, support, step * weight)

    def qdrift(self, token: Token, args: tuple[Expression, ...]) -> None:
        if len(args) != 4:
            reject(
                token,
                "signature",
                "qdrift requires an operator, time, samples and target",
            )
        operator, time_arg, count_arg, target_arg = args
        if (
            count_arg.children
            or not count_arg.token.text.isdecimal()
            or not 1 <= int(count_arg.token.text) <= MAX_QDRIFT_SAMPLES
        ):
            reject(
                count_arg.token,
                "evolution",
                f"qdrift requires a literal sample count in 1..{MAX_QDRIFT_SAMPLES}",
            )
        count = int(count_arg.token.text)
        register, terms = self.pauli_terms(token, operator, target_arg)
        time = self.evolution_angle(time_arg, token)
        weights = [abs(coefficient) for _, coefficient in terms]
        total = sum(weights)
        if not total or not time:
            self.effect_op(
                "qreg.gate", token, (register,), gate="rz", index=0, angle=0.0
            )
            return
        angle = time * total / count
        for support, coefficient in random.Random(QDRIFT_SEED).choices(
            terms, weights=weights, k=count
        ):
            self.pauli_exponential(
                token, register, support, angle if coefficient > 0 else -angle
            )

    def pauli_terms(
        self, token: Token, operator: Expression, target_arg: Expression
    ) -> tuple[Value, list[tuple[tuple[tuple[int, str], ...], float]]]:
        target = self.atom(target_arg)
        register = self.register(target, allow_qubit_array=True)
        width = register_width(register)
        if not operator.children and operator.token.text != "[":
            if operator.token.text not in self.constants:
                reject(operator.token, "pauli", "unknown SparsePauliOp constant")
            operator = self.constants[operator.token.text]
        sparse = operator.token.text == "SparsePauliOp"
        terms = operator
        if sparse:
            terms, width_arg = operator.children
            if _sparse_width(width_arg) != width:
                reject(
                    width_arg.token,
                    "pauli",
                    "num_qubits must match the target array width",
                )
        kind = self.parameters[target.text].kind
        if not terms.children or (kind != "qbit[]" and not (sparse and kind == "qnum")):
            reject(
                token,
                "evolution",
                "evolution requires a nonempty Pauli sum and a qubit array or sparse qnum target",
            )
        prepared = []
        for term in terms.children:
            axes, coefficient = term.children
            support = _pauli_support(axes, width, sparse=sparse)
            prepared.append((support, self.evolution_angle(coefficient, token)))
        return register, prepared

    def pauli_exponential(
        self,
        token: Token,
        register: Value,
        support: tuple[tuple[int, str], ...],
        angle: float,
    ) -> None:
        def gate(name: str, index: int, **attributes: object) -> None:
            self.effect_op(
                "qreg.gate", token, (register,), gate=name, index=index, **attributes
            )

        if not support:
            gate("rz", 0, angle=0.0)
            self.effect_op("qreg.phase", token, (), angle=-angle)
            return

        for index, axis in support:
            if axis == "X":
                gate("h", index)
            elif axis == "Y":
                gate("rx", index, angle=math.pi / 2)
        anchor = support[0][0]
        for index, _ in support[1:]:
            self.effect_op(
                "qreg.pair_gate",
                token,
                (register, register),
                gate="cx",
                indices=(index, anchor),
            )
        gate("rz", anchor, angle=2 * angle)
        for index, _ in reversed(support[1:]):
            self.effect_op(
                "qreg.pair_gate",
                token,
                (register, register),
                gate="cx",
                indices=(index, anchor),
            )
        for index, axis in reversed(support):
            if axis == "X":
                gate("h", index)
            elif axis == "Y":
                gate("rx", index, angle=-math.pi / 2)

    def prepare_state_vector(self, token: Token, args: tuple[Expression, ...]) -> None:
        if len(args) != 3:
            reject(
                token,
                "signature",
                f"{token.text} requires a vector, bound and output",
            )
        vector, bound_arg, output = args
        count = len(vector.children)
        if not 2 <= count <= MAX_PREPARED_AMPLITUDES or count & (count - 1):
            reject(
                vector.token,
                "amplitudes",
                "amplitude count must be a power of two in 2..256",
            )

        def literal(node: Expression) -> float:
            if node.token.text == "/":
                numerator, denominator = (
                    int(part.token.text) for part in node.children
                )
                if not denominator:
                    reject(
                        node.children[1].token,
                        "probabilities",
                        "fraction denominator must be positive",
                    )
                return numerator / denominator
            negative = node.token.text == "-"
            value = node.children[0] if negative else node
            if value.children or not re.fullmatch(
                r"[0-9]+(?:\.[0-9]+)?", value.token.text
            ):
                reject(value.token, "amplitudes", "expected a real decimal literal")
            return float(value.token.text) * (-1 if negative else 1)

        amplitudes = tuple(literal(item) for item in vector.children)
        bound = literal(bound_arg)
        if token.text == "prepare_state":
            if any(value < 0 for value in amplitudes) or not math.isclose(
                math.fsum(amplitudes), 1.0, rel_tol=0, abs_tol=1e-12
            ):
                reject(
                    vector.token,
                    "probabilities",
                    "probabilities must be nonnegative and sum to one",
                )
            if bound < 0:
                reject(
                    bound_arg.token,
                    "probabilities",
                    "probability bound must be nonnegative",
                )
            amplitudes = tuple(math.sqrt(value) for value in amplitudes)
            # The shared preparation is exact, independent of the probability tolerance.
            bound = 0.0
        target = self.atom(output)
        width = count.bit_length() - 1
        destination = self.register(target, width, allow_qubit_array=True)
        if register_width(destination) != width:
            reject(target, "width", "amplitude vector must match the output width")
        self.effect_op(
            "qreg.prepare_amplitudes",
            token,
            (destination,),
            amplitudes=amplitudes,
            bound=bound,
        )

    def xor_update(
        self, token: Token, destination: Expression, source: Expression
    ) -> None:
        target = self.atom(destination)
        register = self.register(target)
        if self.local_scope is not None or self.parameters[target.text].kind != "qnum":
            reject(
                token,
                "statement",
                "in-place XOR requires an integer output at function scope",
            )
        width = register_width(register)
        magnitude = source.children[0] if source.token.text == "-" else source
        if not magnitude.children and magnitude.token.text.isdecimal():
            mask = int(magnitude.token.text)
            if source.token.text == "-":
                if target.text not in self.signed or mask > 1 << (width - 1):
                    reject(
                        source.token,
                        "width",
                        "negative XOR mask must fit the signed target",
                    )
                mask = (-mask) & ((1 << width) - 1)
            if mask >= 1 << width:
                reject(source.token, "width", "XOR mask must fit the target width")
            for bit in range(width):
                if mask & (1 << bit):
                    self.effect_op("qreg.gate", token, (register,), gate="x", index=bit)
            if not mask:
                # The verifier must still check that the target is writable and initialized.
                self.effect_op(
                    "qreg.gate", token, (register,), gate="rz", index=0, angle=0.0
                )
            return
        other = self.atom(source)
        operand = self.register(other)
        operand_width = register_width(operand)
        if (
            operand == register
            or operand_width > width
            or self.parameters[other.text].kind != "qnum"
        ):
            reject(
                source.token,
                "signature",
                "XOR source must be a distinct integer fitting the target",
            )
        count = width if other.text in self.signed else operand_width
        for bit in range(count):
            self.effect_op(
                "qreg.pair_gate",
                token,
                (operand, register),
                gate="cx",
                indices=(min(bit, operand_width - 1), bit),
            )

    def statement(self, statement: Statement, index: str) -> None:
        token, args = statement.token, statement.arguments
        if statement.repeat_index is not None:
            self.repeat(statement, index)
        elif statement.declaration is not None:
            parameter = statement.declaration
            name = parameter.name.text
            if name in self.parameters or name in _INTRINSICS or name in self.constants:
                reject(
                    parameter.name,
                    "name",
                    "local names cannot shadow parameters, locals or intrinsics",
                )
            self.parameters[name] = parameter
            self.widths[name] = parameter.width
        elif statement.control_body is not None:
            self.scalar_control(statement, index)
        elif statement.within is not None:
            self.within(statement, index)
        elif token.text == "=":
            target = self.atom(args[0])
            if "." in args[1].token.text and not args[1].children:
                reject(
                    args[1].token,
                    "operator",
                    "decimal assignment requires min/max, subtraction or multiplication",
                )
            if (
                args[1].token.text == "-"
                and len(args[1].children) == 1
                and target.text in self.parameters
                and self.parameters[target.text].kind == "qbit"
            ):
                reject(
                    args[1].token, "expression", "negation requires a numeric output"
                )
            expression = _Expression(
                args[1], self.widths, f"{self.scope}.expr{index}", self.signed
            )
            if self.local_scope is not None and not expression.quantum:
                reject(
                    token,
                    "scope",
                    "within preparation requires a quantum-dependent expression",
                )
            inferred_width = (
                expression.word_widths[args[1]]
                if args[1].token.text in {"%", "-"}
                else unsigned_width(expression.bound)
            )
            destination = self.register(target, inferred_width)
            if args[1] in expression.signed_words:
                if (
                    self.scope != "main"
                    or self.local_scope is not None
                    or self.parameters[target.text].kind != "qnum"
                    or register_width(destination) != expression.word_widths[args[1]]
                ):
                    reject(
                        target,
                        "type",
                        "signed results require a matching-width main output",
                    )
                self.signed.add(target.text)

            fractional_bits = expression.fractional_bits.get(args[1], 0)
            if fractional_bits:
                if (
                    self.scope != "main"
                    or self.local_scope is not None
                    or self.parameters[target.text].kind != "qnum"
                    or self.parameters[target.text].width is not None
                ):
                    reject(
                        target,
                        "precision",
                        "fractional results require an inferred-width main qnum output",
                    )
            if expression.quantum:
                graph = expression.function()
                self.functions.append(graph)
                inputs = tuple(
                    self.register(self.parameters[name].name)
                    for name in expression.names
                )
                self.effect_op(
                    "qreg.compute", token, (destination, *inputs), callee=graph.name
                )
            else:
                self.effect_op(
                    "qreg.initialize", token, (destination,), value=expression.bound
                )
            if fractional_bits:
                self.fractional_bits[target.text] = fractional_bits
        elif token.text == "phase":
            if len(args) != 2 or "phase" in self.parameters:
                reject(
                    token,
                    "signature",
                    "phase requires an expression and unshadowed pi coefficient",
                )
            coefficient = args[1]
            angle = None
            if "." in coefficient.token.text:
                angle = _literal_angle(coefficient, token)
            elif (
                coefficient.children
                or coefficient.token.text != "pi"
                or "pi" in self.parameters
            ):
                reject(
                    coefficient.token,
                    "coefficient",
                    "phase requires unshadowed pi or a decimal literal",
                )
            expression = _Expression(
                args[0], self.widths, f"{self.scope}.phase{index}", self.signed
            )
            if (
                args[0] in expression.signed_words
                or expression.fractional_bits.get(args[0], 0)
                or not expression.quantum
                or (angle is None and unsigned_width(expression.bound) != 1)
            ):
                reject(
                    args[0].token,
                    "phase",
                    "phase requires a quantum expression and pi requires a one-bit result",
                )
            graph = expression.function()
            self.functions.append(graph)
            inputs = tuple(
                self.register(self.parameters[name].name) for name in expression.names
            )
            phase_attributes = {} if angle is None else {"angle": angle}
            self.effect_op(
                "qreg.phase", token, inputs, callee=graph.name, **phase_attributes
            )
        elif token.text == "^=":
            self.xor_update(token, args[0], args[1])
        elif token.text in {"encode_in_angle", "encode_on_bloch"}:
            self.encode_data(token, args)
        elif token.text == "allocate":
            if len(args) not in {1, 2, 4}:
                reject(
                    token,
                    "signature",
                    "allocate accepts an output, size and output, or size, signedness, zero and output",
                )
            size = None
            if len(args) > 1:
                if args[0].children or not args[0].token.text.isdecimal():
                    reject(
                        args[0].token,
                        "width",
                        "allocation size must be a literal in 1..256",
                    )
                size = int(args[0].token.text)
                if not 1 <= size <= 256:
                    reject(
                        args[0].token,
                        "width",
                        "allocation size must be a literal in 1..256",
                    )
            target = self.atom(args[-1])
            destination = self.register(target, size, allow_qubit_array=True)
            if size is not None and self.widths[target.text] != size:
                reject(
                    target,
                    "width",
                    "allocation size must agree with the declared width",
                )
            if len(args) == 4:
                sign, fraction = args[1:3]
                if (
                    sign.children
                    or sign.token.text not in {"True", "False", "SIGNED", "UNSIGNED"}
                    or fraction.children
                    or fraction.token.text != "0"
                    or self.parameters[target.text].kind != "qnum"
                ):
                    reject(
                        token,
                        "type",
                        "typed allocate requires a qnum, literal signedness and zero fractional bits",
                    )
                if sign.token.text in {"True", "SIGNED"}:
                    if self.scope != "main" or self.local_scope is not None:
                        reject(
                            token,
                            "scope",
                            "signed allocation is supported only at main scope",
                        )
                    self.signed.add(target.text)
            self.effect_op("qreg.allocate", token, (destination,))
        elif token.text in {"prepare_amplitudes", "prepare_state"}:
            self.prepare_state_vector(token, args)
        elif token.text == "qft":
            if len(args) != 1:
                reject(token, "signature", "qft requires one qubit array")
            target = self.atom(args[0])
            register = self.register(target, allow_qubit_array=True)
            if self.parameters[target.text].kind != "qbit[]":
                reject(target, "type", "qft requires a qubit array")
            self.effect_op("qreg.qft", token, (register,))
        elif token.text == "qct_qst_type1":
            if len(args) != 1:
                reject(
                    token, "signature", "Type I transform requires one data register"
                )
            data = self.register(self.atom(args[0]), allow_qubit_array=True)
            self.effect_op("qreg.type1_transform", token, (data,))
        elif token.text in {"qct_type2", "qst_type2", "qct_qst_type2"}:
            kind = {
                "qct_type2": "cosine",
                "qst_type2": "sine",
                "qct_qst_type2": "combined",
            }[token.text]
            if len(args) != 1 + (kind == "combined"):
                reject(
                    token,
                    "signature",
                    "Type II transform requires a data register and, for the combined form, a selector",
                )
            data = self.register(self.atom(args[0]), allow_qubit_array=True)
            selectors: tuple[Value, ...] = ()
            if kind == "combined":
                self.atom(args[1])
                selector, _ = self.gate_target(args[1])
                selectors = (selector,)
            self.effect_op("qreg.type2_transform", token, (data, *selectors), kind=kind)
        elif token.text == "prepare_bell_state":
            if len(args) != 2:
                reject(
                    token, "signature", "prepare_bell_state requires a state and output"
                )
            state, output = args
            if state.children or state.token.text not in {"1", "2"}:
                reject(
                    state.token,
                    "prepare",
                    "only literal Bell states 1 and 2 are admitted",
                )
            target = self.atom(output)
            if (
                target.text not in self.parameters
                or self.parameters[target.text].kind != "qbit[]"
            ):
                reject(
                    target, "type", "prepare_bell_state requires a qubit array output"
                )
            destination = self.register(target, 2, allow_qubit_array=True)
            if self.widths[target.text] != 2:
                reject(target, "width", "Bell preparation requires exactly two qubits")
            self.effect_op("qreg.prepare_ghz", token, (destination,))
            self.effect_op(
                "qreg.gate",
                token,
                (destination,),
                gate="z" if state.token.text == "1" else "x",
                index=0,
            )
        elif token.text == "prepare_ghz_state":
            if len(args) != 2:
                reject(
                    token, "signature", "prepare_ghz_state requires a size and output"
                )
            size_arg, output = args
            if size_arg.children or not size_arg.token.text.isdecimal():
                reject(size_arg.token, "width", "GHZ size requires a literal in 1..256")
            size = int(size_arg.token.text)
            if not 1 <= size <= 256:
                reject(size_arg.token, "width", "GHZ size requires a literal in 1..256")
            target = self.atom(output)
            if (
                target.text not in self.parameters
                or self.parameters[target.text].kind != "qbit[]"
            ):
                reject(
                    target, "type", "prepare_ghz_state requires a qubit array output"
                )
            destination = self.register(target, size, allow_qubit_array=True)
            if self.widths[target.text] != size:
                reject(target, "width", "GHZ size must agree with the output width")
            self.effect_op("qreg.prepare_ghz", token, (destination,))
        elif token.text in _UNIFORM_CALLS:
            trimmed = token.text == "prepare_uniform_trimmed_state"
            if len(args) != (2 if trimmed else 3):
                reject(token, "signature", "invalid uniform preparation arity")
            bounds = args[:-1]
            if any(
                bound.children or not bound.token.text.isdecimal() for bound in bounds
            ):
                reject(
                    token,
                    "signature",
                    "uniform preparation requires literal integer bounds",
                )
            start = 0 if trimmed else int(bounds[0].token.text)
            stop = int(bounds[-1].token.text)
            destination = self.register(self.atom(args[-1]))
            self.effect_op(
                "qreg.uniform_interval", token, (destination,), start=start, stop=stop
            )
        elif token.text == "single_pauli_exponent":
            self.single_pauli_exponent(token, args)
        elif token.text == "suzuki_trotter":
            self.suzuki_trotter(token, args)
        elif token.text == "exponentiate":
            self.exponentiate(token, args)
        elif token.text == "qdrift":
            self.qdrift(token, args)
        elif token.text == "RZZ":
            if len(args) != 2:
                reject(token, "signature", "RZZ requires an angle and a qubit array")
            angle = self.angle(args[0], token)
            target = self.atom(args[1])
            destination = self.register(target, allow_qubit_array=True)
            if (
                self.parameters[target.text].kind != "qbit[]"
                or self.widths[target.text] != 2
            ):
                reject(target, "type", "RZZ requires a two-qubit array")
            self.effect_op("qreg.rzz", token, (destination,), angle=angle)
        elif token.text == "U":
            if len(args) != 5:
                reject(token, "signature", "U requires four angles and a qbit target")
            theta, phi, lam, gamma = (self.angle(arg, token) for arg in args[:4])
            target = self.atom(args[4])
            destination = self.register(target)
            if self.parameters[target.text].kind != "qbit":
                reject(target, "type", "U requires a scalar qbit target")
            for gate, angle in euler_rotation(theta, phi, lam):
                self.effect_op(
                    "qreg.gate", token, (destination,), gate=gate, index=0, angle=angle
                )
            self.effect_op("qreg.phase", token, (), angle=gamma)
        elif token.text == "R":
            if len(args) != 3:
                reject(token, "signature", "R requires two angles and a qbit target")
            theta, phi = (self.angle(arg, token) for arg in args[:2])
            target = self.atom(args[2])
            destination = self.register(target)
            if self.parameters[target.text].kind != "qbit":
                reject(target, "type", "R requires a scalar qbit target")
            for gate, angle in equatorial_rotation(theta, phi):
                self.effect_op(
                    "qreg.gate", token, (destination,), gate=gate, index=0, angle=angle
                )
        elif token.text in ANGLE_CALLS or token.text in _PAIR_CALLS:
            pair = token.text in _PAIR_CALLS
            angled = token.text in ANGLE_CALLS
            if len(args) != (2 if pair else 1) + angled:
                reject(
                    token,
                    "signature",
                    f"{token.text} has the wrong number of arguments",
                )
            attributes: dict[str, object] = {"gate": token.text.lower()}
            if angled:
                attributes["angle"] = self.angle(args[0], token)
            targets = args[int(angled) :]
            locations = tuple(self.gate_target(arg) for arg in targets)
            registers, indices = zip(*locations, strict=True)
            if pair and any(arg.token.text == "[" for arg in targets):
                attributes["indices"] = indices
            if not pair:
                attributes["index"] = indices[0]
            self.effect_op(
                "qreg.pair_gate" if pair else "qreg.gate",
                token,
                registers,
                **attributes,
            )
        elif token.text in _GATE_CALLS:
            if len(args) != 1:
                reject(token, "signature", f"{token.text} requires one output variable")
            if token.text == "hadamard_transform":
                target = self.atom(args[0])
                destination = self.register(target, allow_qubit_array=True)
                width = self.widths[target.text]
                assert width is not None
                bits = range(width)
            else:
                destination, bit = self.gate_target(args[0])
                bits = range(bit, bit + 1)
            for bit in bits:
                self.effect_op(
                    "qreg.gate",
                    token,
                    (destination,),
                    gate=(
                        "h"
                        if token.text == "hadamard_transform"
                        else token.text.lower()
                    ),
                    index=bit,
                )
        else:
            self.call(token, args)

    def adapt(self) -> HybridProgram:
        for index, statement in enumerate(self.syntax.statements):
            self.statement(statement, str(index))
        output_parameters = tuple(
            p for p in self.syntax.parameters if p.role == "output"
        )
        borrowed = tuple(
            self.register(p.name) for p in self.syntax.parameters if p.role == "const"
        )
        declared = tuple(
            self.register(p.name, allow_qubit_array=True, allow_fractional_output=True)
            for p in output_parameters
        )
        declarations = tuple(
            Operation(
                "qreg.declare",
                results=(value,),
                attributes={"name": parameter.name.text},
                location=parameter.name.location,
            )
            for parameter, value in zip(output_parameters, declared, strict=True)
        )
        program = HybridProgram(
            self.scope,
            Region(
                (
                    Block(
                        (self.entry, *borrowed),
                        (
                            *declarations,
                            *self.operations,
                            Operation(
                                "qreg.return",
                                (self.effect, *borrowed, *declared),
                                location=self.syntax.name.location,
                            ),
                        ),
                    ),
                )
            ),
            location=self.syntax.name.location,
        )
        return program


def check_source_version(source_version: str, token: Token) -> None:
    if source_version != SUPPORTED_SOURCE_VERSION:
        reject(
            token,
            "version",
            f"only the pinned source profile {SUPPORTED_SOURCE_VERSION} is admitted",
        )


def _bound_helper(syntax: Function) -> bool:
    return any(port.role in {"real", "inout"} for port in syntax.parameters)


def parse_qmod(
    source: str,
    *,
    filename: str = "<qmod>",
    source_version: str,
    classical_inputs: Mapping[str, float | Sequence[float]] | None = None,
) -> QmodImport:
    """Adapt the pinned unsigned scalar native subset to verified owned records.

    The caller declares source_version; this does not authenticate source bytes.
    Originals are independently checked against their retained source manifest.
    No files, SDKs, synthesis, execution, eval or generated Python are involved.
    """
    tokens = tokenize(source, filename)
    check_source_version(source_version, tokens[0])
    definitions: dict[str, Function] = {}
    functions_syntax, declarations = Parser(tokens).parse()
    constants: dict[str, Expression] = {}
    for name_token, operator in declarations:
        name = name_token.text
        if name in constants or name in _INTRINSICS or name == "main":
            reject(name_token, "name", "duplicate or reserved SparsePauliOp constant")
        terms, width_arg = operator.children
        width = _sparse_width(width_arg)
        if not terms.children:
            reject(
                terms.token,
                "pauli",
                "SparsePauliOp constant requires a nonempty Pauli sum",
            )
        for term in terms.children:
            axes, coefficient = term.children
            _pauli_support(axes, width, sparse=True)
            if coefficient.token.text == "-":
                coefficient = coefficient.children[0]
            _literal_angle(coefficient, name_token)
        constants[name] = operator
    for syntax in functions_syntax:
        name = syntax.name.text
        if name in definitions or name in _INTRINSICS:
            reject(syntax.name, "name", "duplicate function or intrinsic redefinition")
        if name in constants:
            reject(syntax.name, "name", "function shadows a SparsePauliOp constant")
        for parameter in syntax.parameters:
            if parameter.name.text in constants:
                reject(
                    parameter.name, "name", "parameter shadows a SparsePauliOp constant"
                )
        definitions[name] = syntax
    if "main" not in definitions:
        reject(tokens[0], "entry", "source requires one qfunc main")
    main = definitions["main"]
    parameters = {p.name.text: p for p in main.parameters if p.role == "real"}
    inputs = {} if classical_inputs is None else classical_inputs
    if not isinstance(inputs, Mapping) or set(inputs) != set(parameters):
        reject(
            main.name, "binding", f"classical inputs must match {sorted(parameters)}"
        )
    bindings: dict[str, float] = {}
    arrays: dict[str, tuple[float, ...]] = {}

    def checked_real(value: object, name: str) -> float:
        if (
            type(value) not in {float, int}
            or not isinstance(value, (float, int))
            or abs(value) > 2**53
            or not math.isfinite(value)
        ):
            reject(
                main.name,
                "binding",
                f"classical input {name!r} must be a finite real in ±2**53",
            )
        return float(value)

    for name, value in inputs.items():
        array_width = parameters[name].width
        if array_width is None:
            bindings[name] = checked_real(value, name)
        else:
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != array_width
            ):
                reject(
                    main.name,
                    "binding",
                    f"classical input {name!r} requires exactly {array_width} real elements",
                )
            arrays[name] = tuple(checked_real(item, name) for item in value)

    for name, syntax in definitions.items():
        if name == "main" or not _bound_helper(syntax):
            continue
        if len({port.name.text for port in syntax.parameters}) != len(
            syntax.parameters
        ):
            reject(syntax.name, "name", "duplicate helper parameter")
        if any(port.role not in {"real", "inout"} for port in syntax.parameters):
            reject(
                syntax.name,
                "role",
                "bound helpers require real or writable qubit array ports",
            )
        for statement in syntax.statements:
            if (
                statement.declaration is not None
                or statement.token.text not in _GATE_CALLS | _PAIR_CALLS | ANGLE_CALLS
            ):
                reject(statement.token, "scope", "bound helper bodies admit only gates")

    compiled: dict[str, tuple[_Adapt, HybridProgram]] = {}
    active: set[str] = set()

    def resolve(token: Token) -> tuple[Function, HybridProgram]:
        name = token.text
        if name == "main" or name not in definitions:
            reject(token, "call", f"unsupported native helper call {name!r}")
        adapter, program = build(definitions[name])
        return adapter.syntax, program

    def build(syntax: Function) -> tuple[_Adapt, HybridProgram]:
        name = syntax.name.text
        if name in active:
            reject(
                syntax.name, "recursion", "quantum helper call graph must be acyclic"
            )
        if name not in compiled:
            active.add(name)
            adapter = _Adapt(syntax, resolve, definitions, constants)
            if name == "main":
                adapter.classical_values.update(bindings)
                adapter.classical_arrays.update(arrays)
            program = adapter.adapt()
            compiled[name] = adapter, program
            active.remove(name)
        return compiled[name]

    # Bound helpers are checked structurally above and specialized at call sites.
    for syntax in definitions.values():
        if syntax is main or not _bound_helper(syntax):
            build(syntax)
    adapter, program = compiled["main"]
    records: dict[str, HybridProgram] = {}
    for name, (producer, model) in compiled.items():
        for function in producer.functions:
            records[function.name] = function
        if name != "main":
            records[name] = model
    functions = tuple(records.values())
    try:
        verify_register_program(program, functions)
    except HybridVerificationError as error:
        loc = program.location
        assert loc is not None
        diagnostics = tuple(
            (
                item
                if item.message.startswith(f"{loc.filename}:")
                else Diagnostic(
                    item.code, f"{loc.filename}:{loc.line}:{loc.column}: {item.message}"
                )
            )
            for item in error.diagnostics
        )
        raise HybridVerificationError(diagnostics) from error
    return QmodImport(
        program,
        functions,
        source_version,
        MappingProxyType(adapter.inferred),
        filename,
        MappingProxyType(adapter.fractional_bits),
        tuple(p.name.text for p in main.parameters if p.name.text in adapter.signed),
    )
