"""Numeric and primitive Qmod adapter contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from operator import and_, eq, ge, gt, le, lt, ne, or_, xor
from types import MappingProxyType

from ...compiler import _pauli_schedule
from ...compiler._construction.profile import uint_type
from ...compiler._construction.ranges import bitwise_width, unsigned_width
from ...compiler._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    Value,
    ValueId,
)
from .angles import REAL_LITERAL
from .cscope import QaoaWorkflow
from .fixed_payoff import inferred_computation
from .syntax import (
    ANGLE_CALLS,
    EXTREMA,
    Expression,
    Function,
    Token,
    reject,
)

SUPPORTED_SOURCE_VERSION = "2bfd835e73b6882fcc515a725098b04766c238e7"

MAX_EFFECT_OPERATIONS = 4096

MAX_HELPER_EXPANSIONS = 4096

MAX_LOOP_ITERATIONS = 4096

MAX_QDRIFT_SAMPLES = _pauli_schedule.MAX_QDRIFT_SAMPLES

QDRIFT_SEED = _pauli_schedule.DEFAULT_QDRIFT_SEED

_GATE_CALLS = frozenset({"X", "Y", "H", "S", "Z", "I", "SX", "hadamard_transform"})

_PAIR_CALLS = frozenset({"CX", "CZ", "SWAP", "CRX", "CRY"})

_STATIC_COMPARISONS = {"==": eq, "!=": ne, "<": lt, "<=": le, ">": gt, ">=": ge}

_STATIC_GATES = (
    _GATE_CALLS
    | _PAIR_CALLS
    | ANGLE_CALLS
    | {"IDENTITY", "assign_phase_table", "RESET"}
)

_UNIFORM_CALLS = frozenset(
    {"prepare_uniform_interval_state", "prepare_uniform_trimmed_state"}
)

_INTRINSICS = (
    _UNIFORM_CALLS
    | _GATE_CALLS
    | _PAIR_CALLS
    | {
        "allocate",
        "RESET",
        "apply_to_all",
        "encode_in_angle",
        "encode_on_bloch",
        "inplace_encode_on_bloch",
        "phase",
        "grover_diffuser",
        "qsvt_inversion",
        "qsvt_lcu",
        "multiswap",
        "pi",
        "control",
        "prepare_ghz_state",
        "prepare_exponential_state",
        "inplace_prepare_state",
        "inplace_prepare_amplitudes",
        "IDENTITY",
        "exact_amplitude_amplification",
        "amplitude_amplification",
        "prepare_amplitudes",
        "prepare_state",
        "prepare_bell_state",
        "suzuki_trotter",
        "multi_suzuki_trotter",
        "single_pauli_exponent",
        "linear_pauli_rotations",
        "select_rotation",
        "unitary",
        "exponentiate",
        "qdrift",
        "qaoa_penalty",
        "full_hea",
        "qft",
        "swap_test",
        "drop",
        "free",
        "modular_exp",
        "modular_multiply_constant_inplace",
        "qpe_flexible",
        "qpe",
        "qct_qst_type1",
        "qct_type2",
        "qst_type2",
        "qct_qst_type2",
        "grover_operator",
        "grover_search",
        "phase_oracle",
    }
    | EXTREMA
    | ANGLE_CALLS
)


def _literal_angle(angle: Expression, gate: Token) -> float:
    if (
        angle.children
        or not REAL_LITERAL.fullmatch(angle.token.text)
        or Decimal(angle.token.text) > 2**53
    ):
        reject(
            angle.token,
            "angle",
            f"{gate.text} angle must be a literal in 0..2**53 radians",
        )
    return float(angle.token.text)


def _preparation_value(node: Expression) -> float:
    if node.token.text == "sqrt":
        radicand = _preparation_value(node.children[0])
        if radicand < 0 or not math.isfinite(radicand):
            reject(
                node.token,
                "amplitudes",
                "square root requires a finite nonnegative literal",
            )
        return math.sqrt(radicand)
    if node.token.text == "/":
        numerator, denominator = (int(part.token.text) for part in node.children)
        if not denominator:
            reject(
                node.children[1].token,
                "probabilities",
                "fraction denominator must be positive",
            )
        return numerator / denominator
    negative = node.token.text == "-"
    value = node.children[0] if negative else node
    if value.children or not REAL_LITERAL.fullmatch(value.token.text):
        reject(value.token, "amplitudes", "expected a real decimal literal")
    return float(value.token.text) * (-1 if negative else 1)


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
    workflow: QaoaWorkflow | None = None
    deferred_measurements: bool = False
    exact_mps: bool = False


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


def _bound_helper(syntax: Function) -> bool:
    return (
        (
            len(syntax.statements) == 1
            and syntax.statements[0].token.text == "$kickback_phase"
        )
        or inferred_computation(syntax) is not None
        or any(
            port.role in {"real", "integer", "inout", "callable", "operator"}
            or port.kind == "qphase"
            or port.fractional_bits
            for port in syntax.parameters
        )
    )
