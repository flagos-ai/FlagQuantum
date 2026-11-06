"""Native syntax to the admitted Compiler models, without synthesis or execution."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from itertools import groupby
from operator import and_, eq, ge, gt, le, lt, ne, or_, xor
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
from .affine import lower_affine_update, lower_snapshot, snapshot_start
from .amplification import expand_amplification, lower_amplification
from .angles import REAL_LITERAL
from .array_kickback import expand_nested_array_kickbacks
from .basis_regions import emit_closed_region, try_closed_region
from .bindings import (
    is_binding_scope,
    lower_binding_scope,
    publish_bound_array,
    publish_partition,
    record_view,
)
from .borrowed_phase_views import expand_borrowed_phase_views
from .callback_arrays import (
    capture_callbacks,
    capture_effects,
    indexed_callback,
    lower_callback_control,
    lower_selected_effects,
)
from .clean_selector_normalization import normalize_clean_selectors, probe_source
from .cnf_kickback import lower_cnf_kickback
from .computed_rotation import (
    composed_rotation,
    computed_output,
    lower_computed_rotation,
    reinterpreted_selector,
)
from .consuming_preparation import fuse_consuming_preparations
from .controlled_evolution import EVOLUTION_CALLS, controlled_evolution
from .copied_selector import lower_copied_selector, slice_output
from .cscope import QaoaWorkflow, VqeSettings, split_cscope
from .decisions import lower_decision_phase
from .deferred_measurement import defer_private_measurements
from .diffusion import lower_diffuser
from .fixed_payoff import inferred_computation, lower_fixed_payoff, payoff_helpers
from .fixed_regions import (
    FixedPreparation,
    build_fixed_preparation,
    emit_fixed_preparation,
)
from .fixed_xor import lower_fixed_xor
from .hea import lower_hea
from .indexed_control import (
    lower_constant_phase_guard,
    lower_guarded_field_increment,
    lower_indexed_control,
    lower_nested_indexed_control,
    lower_zero_reflection_control,
)
from .indexed_pauli import bind_indexed_pauli, indexed_pauli_member
from .integers import checked_integer, checked_integer_array, integer_operation
from .iterations import expand_iterations
from .local_cleanup import lower_free
from .masked_selection import (
    emit_selected,
    emit_signed_predicate_xor,
    lower_masked_selection,
)
from .matrices import lower_unitary, validate_matrix_constant
from .oracles import expand_static_oracles
from .parity_updates import lower_indexed_parity, lower_masked_word_xor
from .pauli_schedule import ceiling_repetitions, compact_commuting
from .permutations import is_const_cx_permutation, verify_permutations
from .phase import MAX_PHASE_INPUT_BITS, expand_phase
from .phase_factors import Cube, preserve_factors
from .phase_kickback import expand_phase_kickbacks
from .predicates import expand_predicate_oracles
from .preparation import exponential_angles, lower_inplace_preparation
from .qpe import (
    infer_phase_allocation,
    is_modular_power,
    lower_modular_loop,
    lower_modular_qpe,
    lower_qpe,
    power_action,
    specialize_power_body,
)
from .qsvt import lower_qsvt_inversion
from .quantized_numeric import emit_numeric_program, try_numeric_program
from .quantum_callbacks import (
    BORROWED_EFFECT_SIGNATURES,
    CALLBACK_ARRAY,
    Lambda,
    specialize_quantum_callbacks,
)
from .real_matrices import bound_matrix, lower_matrix_iteration, matrix_cell
from .repeated_effects import lower_repeated_effects
from .scope_control import lower_fourier_control
from .selector_rotations import (
    fold_skip_control_tables,
    is_selector_helper,
    lower_guarded_selector,
    lower_select_rotation,
)
from .small_pauli import lower_narrow_pauli_sequence
from .swap_test import lower_swap_test
from .syntax import (
    ANGLE_CALLS,
    EXTREMA,
    Expression,
    Function,
    Parameter,
    Parser,
    Record,
    Statement,
    Token,
    exact_unsigned_literal,
    reject,
    tokenize,
)
from .unary_selection_proof import prove_annotated_selections
from .unitary_effects import (
    PURE_EFFECTS,
    emit_controlled_pauli,
    emit_inverse,
    is_reversible_scope,
    lower_reversible_scope,
    preserves_basis_labels,
)
from .variational import lower_qaoa_penalty

SUPPORTED_SOURCE_VERSION = "2bfd835e73b6882fcc515a725098b04766c238e7"
MAX_EFFECT_OPERATIONS = 4096
MAX_HELPER_EXPANSIONS = 4096
MAX_LOOP_ITERATIONS = 4096
MAX_QDRIFT_SAMPLES = 2048
QDRIFT_SEED = 0
_GATE_CALLS = frozenset({"X", "Y", "H", "S", "Z", "I", "SX", "hadamard_transform"})
_PAIR_CALLS = frozenset({"CX", "CZ", "SWAP", "CRX"})
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
        record_types: Mapping[str, Record] = MappingProxyType({}),
    ) -> None:
        self.record_types = record_types
        self.syntax = syntax
        self.resolve = resolve
        self.definitions = definitions
        self.constants = constants
        self.classical_values: dict[str, float] = {}
        self.callback_bindings: dict[str, Function] = {}
        self.zero_callbacks: dict[str, tuple[Operation, ...]] = {}
        self.callback_arrays: dict[str, tuple[tuple[Operation, ...], ...]] = {}
        self.classical_arrays: dict[str, tuple[float, ...]] = {}
        self.classical_matrices: dict[str, tuple[tuple[float, ...], ...]] = {}
        self.classical_integers: dict[str, int] = {}
        self.integer_arrays: dict[str, tuple[int, ...]] = {}
        self.pauli_bindings: dict[str, Expression] = {}
        self.indexed_pauli: dict[str, tuple[tuple[int, int], ...]] = {}
        self.scope = syntax.name.text
        self.parameters: dict[str, Parameter] = {}
        for parameter in syntax.parameters:
            name = parameter.name.text
            if name in self.parameters:
                reject(parameter.name, "name", f"duplicate output {name!r}")
            self.parameters[name] = (
                replace(parameter, kind="qnum")
                if parameter.kind == "qnum_signed"
                else parameter
            )
        self.widths = {name: p.width for name, p in self.parameters.items()}
        self.inferred: dict[str, int] = {}
        self.signed: set[str] = {
            p.name.text
            for p in syntax.parameters
            if p.kind in {"qphase_signed", "qnum_signed"}
        }
        self.fractional_bits: dict[str, int] = {}
        self.borrowed_zero_phases: set[Value] = set()
        self.registers: dict[str, Value] = {}
        self.views: dict[str, tuple[tuple[Value, int], ...]] = {}
        self.suspended: set[str] = set()
        self.counter = len(self.parameters) + 1
        self.entry = Value(ValueId(self.scope, 0), QUANTUM_EFFECT)
        self.effect = self.entry
        self.operations: list[Operation] = []
        self.functions: list[HybridProgram] = []
        self.loop_indices: dict[str, int] = {}
        self.local_scope: list[tuple[str, Token]] | None = None
        self.dropped: set[str] = set()
        self.helper_stack: tuple[str, ...] = ()
        self.helper_expansions = 0
        self.loop_iterations = 0
        self.selector_guard: tuple[tuple[tuple[Value, int], ...], str, int] | None = (
            None
        )

    def register(
        self,
        token: Token,
        inferred: int | None = None,
        *,
        allow_qubit_array: bool = False,
        allow_fractional_output: bool = False,
        allocate_local: bool = False,
    ) -> Value:
        name = token.text
        if name in self.suspended:
            reject(token, "binding", "a bound source name is suspended")
        if name in self.views:
            reject(token, "view", "this operation requires a whole register")
        if name in self.dropped:
            reject(token, "drop", "a dropped register cannot be used again")
        if name in self.fractional_bits and not allow_fractional_output:
            reject(token, "precision", "a fractional result is a terminal output")
        if name not in self.parameters:
            reject(token, "name", f"unknown output {name!r}")
        if (
            self.parameters[name].kind in {"qphase", "qphase_signed"}
            and not allow_fractional_output
        ):
            reject(
                token, "precision", "fixed-point phase register requires qpe_flexible"
            )
        if self.parameters[name].kind == "qbit[]" and not allow_qubit_array:
            reject(
                token,
                "type",
                "qubit arrays support allocation and whole-array Hadamard",
            )
        if self.parameters[name].role in {"real", "integer"}:
            reject(token, "type", "classical parameter is not a quantum register")
        local = self.parameters[name].role == "local"
        hidden = local and (
            self.parameters[name].kind == "qbit[]"
            or (
                allocate_local
                and (
                    self.scope == "main"
                    or (self.helper_stack and self.parameters[name].kind == "qbit")
                )
                and self.local_scope is None
                and self.parameters[name].kind in {"qbit", "qnum", "qphase"}
            )
        )
        if (
            local
            and not hidden
            and self.local_scope is None
            and name not in self.registers
        ):
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
                if not hidden:
                    assert self.local_scope is not None
                    self.local_scope.append((name, token))
                self.append_op(
                    token,
                    Operation(
                        "qreg.hidden" if hidden else "qreg.local",
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

    def record_field(
        self, node: Expression, *, allow_fractional: bool = False
    ) -> tuple[Parameter, tuple[tuple[Value, int], ...]] | None:
        if node.token.text != "." or len(node.children) != 2:
            return None
        root, member = node.children
        parent = self.record_field(root) if root.children else None
        parameter = (
            parent[0] if parent is not None else self.parameters.get(root.token.text)
        )
        if parameter is None or parameter.record is None:
            reject(root.token, "field", "field access requires a declared record")
        record = self.record_types[parameter.record]
        offset = 0
        for field in record.fields:
            assert field.width is not None
            size = field.width * (field.length or 1)
            if field.name.text == member.token.text:
                if field.fractional_bits and not allow_fractional:
                    reject(
                        member.token,
                        "precision",
                        "fractional record fields require checked fixed-point lowering",
                    )
                view = parent[1] if parent is not None else self.array_view(root)
                return field, view[offset : offset + size]
            offset += size
        reject(member.token, "field", "unknown record field")

    def record_bit(self, node: Expression) -> tuple[Value, int] | None:
        """Resolve one checked row/column access without erasing field rank."""
        if node.token.text != "[" or node.children[0].token.text != "[":
            return None
        row, column = node.children
        resolved = self.record_field(row.children[0])
        if resolved is None:
            return None
        field, locations = resolved
        if field.kind != "qbit[][]":
            reject(
                node.token,
                "field",
                "double indexing requires a two-dimensional bit field",
            )
        assert field.width is not None and field.length is not None
        row_index = self.static_integer(row.children[1])
        column_index = self.static_integer(column)
        if not 0 <= row_index < field.length:
            reject(row.token, "width", "field row index is out of bounds")
        if not 0 <= column_index < field.width:
            reject(column.token, "width", "field column index is out of bounds")
        return locations[row_index * field.width + column_index]

    def static_integer(self, node: Expression) -> int:
        text = node.token.text
        if text == "$exact_log2":
            size = self.static_integer(node.children[0])
            if size < 1 or size & (size - 1):
                reject(
                    node.token,
                    "width",
                    "static log base two requires a positive power of two",
                )
            return size.bit_length() - 1
        if text == "$ceil_log2":
            size = self.static_integer(node.children[0])
            if size < 2:
                reject(
                    node.token,
                    "width",
                    "binary log width requires a static integer of at least two",
                )
            return (size - 1).bit_length()
        if text.isdecimal():
            return checked_integer(int(text), node.token)
        if not node.children and text in self.classical_integers:
            return self.classical_integers[text]
        if text == "[" and not node.children[0].children:
            values = self.integer_arrays.get(node.children[0].token.text)
            if values is None:
                reject(
                    node.token,
                    "expression",
                    "static indices require an integer array, not quantum or real data",
                )
            offset = self.static_integer(node.children[1])
            if not 0 <= offset < len(values):
                reject(node.token, "binding", "integer array index is out of bounds")
            return values[offset]
        if text == "-" and len(node.children) == 1:
            return checked_integer(-self.static_integer(node.children[0]), node.token)
        if text in {"+", "-", "*", "%", "**"}:
            left, right = (self.static_integer(child) for child in node.children)
            return integer_operation(node.token, left, right)
        if text == "floor" and node.children[0].token.text == "/":
            numerator, denominator = (
                self.static_integer(child) for child in node.children[0].children
            )
            if denominator <= 0:
                reject(
                    node.token, "expression", "static floor requires a positive divisor"
                )
            return numerator // denominator
        if text == ".":
            root = node.children[0]
            member = node.children[1].token.text if len(node.children) == 2 else "len"
            if (entry := indexed_pauli_member(self, node, root, member)) is not None:
                return entry
            if member == "size" and root.children:
                field = self.record_field(root)
                if field is not None:
                    return len(field[1])
            if member == "size" and not root.children:
                parameter = self.parameters.get(root.token.text)
                width = self.widths.get(root.token.text)
                if (
                    parameter is not None
                    and parameter.kind in {"qbit", "qbit[]", "qnum"}
                    and width is not None
                ):
                    return width
            if member != "len":
                reject(node.token, "field", "expected array len or record size")
            if root.children:
                if root.token.text == "[":
                    resolved = self.record_field(root.children[0])
                    if resolved is not None and resolved[0].kind == "qbit[][]":
                        return len(self.array_view(root))
                field = self.record_field(root)
                if field is not None and field[0].kind in {
                    "qbit[]",
                    "qnum[]",
                    "qbit[][]",
                }:
                    return field[0].length or len(field[1])
                reject(node.token, "field", "len requires an array field")
            name = root.token.text
            numeric = self.parameters.get(name)
            if (
                numeric is not None
                and numeric.length is not None
                and numeric.record is None
            ):
                return numeric.length
            if name in self.callback_arrays:
                return len(self.callback_arrays[name])
            if name in self.classical_arrays:
                return len(self.classical_arrays[name])
            if name in self.classical_matrices:
                return len(self.classical_matrices[name])
            if name in self.integer_arrays:
                return len(self.integer_arrays[name])
            if name in self.indexed_pauli:
                return len(self.indexed_pauli[name])
            width = self.widths.get(name)
            if (
                width is None
                or self.parameters[name].kind != "qbit[]"
                or self.parameters[name].record is not None
            ):
                reject(
                    node.token, "width", "len requires a qubit array with a known width"
                )
            return width
        if text in self.loop_indices:
            return self.loop_indices[text]
        reject(node.token, "expression", "expected a static integer or loop index")

    def gate_target(
        self, node: Expression, *, allow_numeric: bool = False
    ) -> tuple[Value, int]:
        cell = self.record_bit(node)
        if cell is not None:
            return cell
        source = node.children[0] if node.token.text == "[" else node
        if source.token.text == ".":
            resolved = self.record_field(source)
            assert resolved is not None
            field, field_view = resolved
            if node.token.text == "[":
                if field.kind != "qbit[]":
                    reject(node.token, "field", "indexing requires an array field")
                index = self.static_integer(node.children[1])
                if not 0 <= index < len(field_view):
                    reject(node.token, "width", "field index is out of bounds")
                return field_view[index]
            if field.kind != "qbit":
                reject(node.token, "field", "gate requires a scalar field or element")
            return field_view[0]
        indexed = node.token.text == "["
        target = self.atom(node.children[0] if indexed else node)
        if target.text in self.suspended:
            reject(target, "binding", "a bound source name is suspended")
        parameter = self.parameters.get(target.text)
        view = self.views.get(target.text)
        if indexed and (parameter is None or parameter.kind != "qbit[]"):
            reject(target, "operator", "indexing requires a qubit array")
        index = self.static_integer(node.children[1]) if indexed else 0
        if indexed and parameter is not None and parameter.length is not None:
            if self.widths[target.text] != parameter.length:
                reject(
                    target,
                    "type",
                    "a scalar gate requires a one-bit numeric array element",
                )
        if view is not None:
            width = len(view)
        else:
            value = self.register(target, allow_qubit_array=True)
            width = register_width(value)
        scalar_kinds = {"qbit", "qbit[]"}
        if (
            allow_numeric
            and target.text not in self.signed
            and target.text not in self.fractional_bits
        ):
            scalar_kinds.add("qnum")
        if not indexed and (
            parameter is None or parameter.kind not in scalar_kinds or width != 1
        ):
            reject(
                target, "type", "gate operand requires a scalar qubit or array element"
            )
        if not 0 <= index < width:
            reject(node.token, "width", "qubit array index is out of bounds")
        if view is not None:
            return view[index]
        return value, index

    def array_view(
        self, node: Expression, *, allow_phase: bool = False, allow_signed: bool = False
    ) -> tuple[tuple[Value, int], ...]:
        if node.token.text == "[" and not node.children[0].children:
            source, index_expression = node.children
            port = self.parameters.get(source.token.text)
            if port is not None and port.length is not None and port.record is None:
                offset = self.static_integer(index_expression)
                if not 0 <= offset < port.length:
                    reject(
                        index_expression.token,
                        "width",
                        "numeric array index is out of bounds",
                    )
                array = self.array_view(source)
                width = len(array) // port.length
                return array[offset * width : (offset + 1) * width]
        cell = self.record_bit(node)
        if cell is not None:
            return (cell,)
        if node.token.text == "[" and node.children[0].token.text == ".":
            resolved = self.record_field(node.children[0])
            assert resolved is not None
            field, word_locations = resolved
            if field.kind not in {"qnum[]", "qbit[][]"}:
                return (self.gate_target(node),)
            index = self.static_integer(node.children[1])
            assert field.length is not None and field.width is not None
            if not 0 <= index < field.length:
                reject(node.token, "width", "field index is out of bounds")
            return word_locations[index * field.width : (index + 1) * field.width]
        if node.token.text == ".":
            resolved = self.record_field(node)
            assert resolved is not None
            if resolved[0].kind == "qnum_signed" and not allow_signed:
                reject(
                    node.token,
                    "type",
                    "signed field requires an explicit signed or raw-bit view",
                )
            return resolved[1]
        if node.token.text == "{":
            return tuple(
                location
                for item in node.children
                for location in (
                    (self.gate_target(item),)
                    if item.token.text == "["
                    else self.array_view(item)
                )
            )
        if node.token.text == ":":
            source, start_node, stop_node = node.children
            source_view = self.array_view(
                source, allow_phase=allow_phase, allow_signed=allow_signed
            )
            start, stop = self.static_integer(start_node), self.static_integer(
                stop_node
            )
            if not 0 <= start < stop <= len(source_view):
                reject(
                    node.token, "width", "qubit array slice is out of bounds or empty"
                )
            return source_view[start:stop]
        target = self.atom(node)
        if target.text in self.suspended:
            reject(target, "binding", "a bound source name is suspended")
        parameter = self.parameters.get(target.text)
        phase = (
            allow_phase
            and parameter is not None
            and (
                parameter.kind == "qphase"
                or (allow_signed and parameter.kind == "qphase_signed")
            )
        )
        if parameter is None or (
            parameter.kind not in {"qbit[]", "qnum", "qbit"} and not phase
        ):
            reject(target, "type", "expected an unsigned quantum register or view")
        if (target.text in self.signed and not allow_signed) or (
            target.text in self.fractional_bits and not phase
        ):
            reject(target, "type", "views require an unsigned integer register")
        view = self.views.get(target.text)
        if view is None:
            value = self.register(
                target, allow_qubit_array=True, allow_fractional_output=phase
            )
            view = tuple((value, index) for index in range(register_width(value)))
        return view

    def integer_array(self, node: Expression) -> tuple[int, ...]:
        if node.token.text == "[]":
            return checked_integer_array(
                tuple(self.static_integer(child) for child in node.children), node.token
            )
        if not node.children and node.token.text in self.integer_arrays:
            return self.integer_arrays[node.token.text]
        reject(
            node.token,
            "binding",
            "expected a classical integer array literal or binding",
        )

    def real_array(self, node: Expression) -> tuple[float, ...]:
        if node.token.text == "[]":
            return tuple(self.classical_number(item) for item in node.children)
        source = node.children[0] if node.token.text == ":" else node
        values = self.classical_arrays.get(source.token.text)
        if values is None or source.children:
            reject(source.token, "binding", "helper requires a bound real array")
        if node.token.text != ":":
            return values
        start, stop = (self.static_integer(item) for item in node.children[1:])
        if not 0 <= start < stop <= len(values):
            reject(node.token, "binding", "real array slice is out of bounds or empty")
        return values[start:stop]

    def repeat(self, statement: Statement, index: str) -> None:
        variable = statement.repeat_index
        assert variable is not None
        name = variable.text
        if (
            name in self.parameters
            or name in _INTRINSICS
            or name in self.constants
            or name in self.loop_indices
        ):
            reject(variable, "name", "loop index cannot shadow a register or intrinsic")
        count = self.static_integer(statement.arguments[0])
        if not 0 <= count <= 256:
            reject(
                statement.token, "budget.expansion", "repeat count must be in 0..256"
            )
        self.loop_iterations += count
        if self.loop_iterations > MAX_LOOP_ITERATIONS:
            reject(
                statement.token,
                "budget.expansion",
                "static loops exceed 4096 iterations",
            )
        if lower_modular_loop(self, statement, count) or lower_repeated_effects(
            self, statement, count
        ):
            return
        for item in statement.repeat_body:
            callee = self.callback_bindings.get(
                item.token.text
            ) or self.definitions.get(item.token.text)
            bound_call = (
                callee is not None
                and callee.name.text != "main"
                and _bound_helper(callee)
            )
            if item.declaration is not None or (
                item.token.text
                not in _STATIC_GATES
                | {
                    "phase",
                    "suzuki_trotter",
                    "multi_suzuki_trotter",
                    "unitary",
                    "apply_to_all",
                }
                and not bound_call
                and item.control_body is None
                and item.callback_index is None
                and item.map_callback is None
                and item.static_if_body is None
                and item.repeat_index is None
                and item.inverse_body is None
                and not is_reversible_scope(item)
            ):
                reject(
                    item.token,
                    "scope",
                    "repeat bodies admit only gates and bound helper calls",
                )
        for iteration in range(count):
            self.loop_indices[name] = iteration
            for ordinal, item in enumerate(statement.repeat_body):
                self.statement(item, f"{index}.{iteration}.{ordinal}")
        self.loop_indices.pop(name, None)

    def classical_number(self, node: Expression) -> float:
        text = node.token.text
        if not node.children:
            if text in self.classical_integers or text in self.loop_indices:
                value = self.static_integer(node)
                if abs(value) > 2**53:
                    reject(
                        node.token,
                        "precision",
                        "integer angle conversion exceeds exact real range",
                    )
                return float(value)
            if text in self.classical_values:
                return self.classical_values[text]
            if text == "pi" and text not in self.parameters:
                return math.pi
            constant = self.constants.get(text)
            if constant is not None and text not in self.parameters:
                return _literal_angle(constant, node.token)
            return _literal_angle(node, node.token)
        if text == "[":
            if (cell := matrix_cell(self, node)) is not None:
                return cell
            source, offset = node.children
            if not source.children and source.token.text in self.integer_arrays:
                value = self.static_integer(node)
                if abs(value) > 2**53:
                    reject(
                        node.token,
                        "precision",
                        "integer angle conversion exceeds exact real range",
                    )
                return float(value)
            values = (
                self.real_array(source)
                if source.token.text == ":"
                else self.classical_arrays.get(source.token.text)
            )
            if (source.children and source.token.text != ":") or values is None:
                reject(source.token, "binding", "expected a bound classical array")
            index = self.static_integer(offset)
            if not 0 <= index < len(values):
                reject(offset.token, "binding", "real array index is out of bounds")
            return values[index]
        if text == ".":
            return float(self.static_integer(node))
        if text == "**":
            value = self.static_integer(node)
            if abs(value) > 2**53:
                reject(
                    node.token,
                    "precision",
                    "integer angle conversion exceeds exact real range",
                )
            result = float(value)
        elif text in {"acos", "asin", "sqrt", "exp"}:
            value_real = self.classical_number(node.children[0])
            if text in {"acos", "asin"}:
                if not -1 <= value_real <= 1:
                    reject(
                        node.token, "angle", f"{text} requires an argument in [-1, 1]"
                    )
                result = (
                    math.acos(value_real) if text == "acos" else math.asin(value_real)
                )
            elif text == "sqrt":
                if value_real < 0:
                    reject(node.token, "angle", "sqrt requires a nonnegative argument")
                result = math.sqrt(value_real)
            else:
                if value_real > 53 * math.log(2):
                    reject(
                        node.token, "angle", "exponential exceeds finite angle range"
                    )
                result = math.exp(value_real)
        elif text == "-" and len(node.children) == 1:
            result = -self.classical_number(node.children[0])
        elif text in {"+", "-", "*", "/"}:
            left, right = (self.classical_number(child) for child in node.children)
            if text == "/" and right == 0:
                reject(node.token, "angle", "classical division by zero")
            if text == "+":
                result = left + right
            elif text == "-":
                result = left - right
            elif text == "*":
                result = left * right
            else:
                result = left / right
        else:
            reject(node.token, "angle", "expected a finite classical real expression")
        if not math.isfinite(result) or abs(result) > 2**53:
            reject(node.token, "angle", "classical real exceeds finite ±2**53 range")
        return result

    def angle(self, node: Expression, gate: Token) -> float:
        if node.token.text in {
            "+",
            "-",
            "*",
            "/",
            "**",
            "pi",
            "acos",
            "asin",
            "sqrt",
            "exp",
        } and (node.children or node.token.text == "pi"):
            return self.classical_number(node)
        if (
            not node.children
            and (
                node.token.text in self.classical_integers
                or node.token.text in self.loop_indices
            )
        ) or (
            node.token.text == "["
            and node.children[0].token.text in self.integer_arrays
        ):
            return self.classical_number(node)
        if node.token.text == "[" and node.children[0].token.text == ":":
            return self.classical_number(node)
        if not node.children and node.token.text in self.classical_values:
            return self.classical_values[node.token.text]
        if node.token.text == "[":
            if (cell := matrix_cell(self, node)) is not None:
                return cell
            array, index = node.children
            values = self.classical_arrays.get(array.token.text)
            if values is not None and not array.children:
                if (
                    not self.helper_stack
                    and not self.loop_indices
                    and (index.children or not index.token.text.isdecimal())
                ):
                    reject(
                        index.token,
                        "binding",
                        "real array index requires an integer literal",
                    )
                offset = self.static_integer(index)
                if not 0 <= offset < len(values):
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
        inplace = token.text == "inplace_encode_on_bloch"
        bloch = token.text != "encode_in_angle"
        width = (len(values) + 1) // 2 if bloch else len(values)
        parameter = self.parameters.get(output.text)
        if parameter is None or (
            parameter.kind != "qbit[]"
            and not (inplace and parameter.kind == "qnum" and width == 1)
        ):
            reject(
                output,
                "type",
                "encoding requires a qubit array or one-bit numeric in-place target",
            )
        if inplace and len(values) % 2:
            reject(
                source,
                "binding",
                "in-place Bloch encoding requires complete data pairs",
            )
        target = self.register(
            output, None if inplace else width, allow_qubit_array=True
        )
        if register_width(target) != width:
            reject(output, "width", "encoding output width must match its data")
        if not inplace:
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

    def borrowed_effect_callback(
        self, argument: Expression, kinds: tuple[str, ...]
    ) -> Function:
        if isinstance(argument, Lambda):
            if len(argument.parameters) != len(kinds) or len(
                {p.text for p in argument.parameters}
            ) != len(kinds):
                reject(
                    argument.token,
                    "callback.signature",
                    (
                        "callback requires two distinct array ports"
                        if len(kinds) == 2
                        else "callback requires one numeric port"
                    ),
                )
            location = argument.token.location
            return Function(
                Token(f"$callback_{location.line}_{location.column}", location),
                tuple(
                    Parameter(port, kind, None, "inout")
                    for port, kind in zip(argument.parameters, kinds, strict=True)
                ),
                argument.body,
            )
        name = self.atom(argument)
        callback = self.callback_bindings.get(name.text) or self.definitions.get(
            name.text
        )
        if (
            callback is None
            or tuple(p.kind for p in callback.parameters) != kinds
            or any(
                p.record is not None or p.role != "inout" for p in callback.parameters
            )
        ):
            reject(
                name,
                "callback.signature",
                "callback requires matching borrowed quantum ports",
            )
        return callback

    def call_bound_helper(
        self, syntax: Function, token: Token, arguments: tuple[Expression, ...]
    ) -> None:
        if len(arguments) != len(syntax.parameters):
            reject(token, "signature", "helper argument count disagrees")
        if any(port.fractional_bits for port in syntax.parameters):
            reject(
                token,
                "precision",
                "fixed-point helpers require a checked computed rotation",
            )
        if any(p.kind == "SparsePauliOp" for p in syntax.parameters):
            reject(
                token,
                "qpe",
                "typed power helpers require a lexical qpe_flexible callback",
            )
        if syntax.name.text in self.helper_stack or len(self.helper_stack) >= 16:
            reject(
                token,
                "recursion",
                "bound helper expansion must be acyclic and at most 16 levels",
            )
        if self.helper_expansions >= MAX_HELPER_EXPANSIONS:
            reject(
                token,
                "budget.expansion",
                "native expansion exceeds 4096 bound helper calls",
            )
        child = _Adapt(
            syntax, self.resolve, self.definitions, self.constants, self.record_types
        )
        child.helper_stack = (*self.helper_stack, syntax.name.text)
        child.helper_expansions = self.helper_expansions + 1
        child.loop_iterations = self.loop_iterations
        child.selector_guard = self.selector_guard
        child.classical_integers.update(
            (name, value)
            for name, value in self.classical_integers.items()
            if name in self.constants
        )
        actuals: set[tuple[Value, int]] = set()
        outputs: set[Value] = set()
        phase_borrows: set[Value] = set()
        readonly: set[tuple[Value, int]] = set()
        modular_phase_ports = {
            statement.arguments[0].token.text
            for statement in syntax.statements
            if is_modular_power(statement)
        }
        phase_exports: dict[str, str] = {}
        signed_bit_exports: dict[str, int] = {}
        deferred: dict[str, Token] = {}
        for argument, parameter in zip(arguments, syntax.parameters, strict=True):
            name = parameter.name.text
            if parameter.kind == "qfunc()":
                if isinstance(argument, Lambda):
                    if argument.parameters or not argument.body:
                        reject(
                            argument.token,
                            "callback.signature",
                            "expected a nonempty zero-argument callback",
                        )
                    effects = capture_effects(self, argument.token, argument.body)
                else:
                    captured = self.zero_callbacks.get(argument.token.text)
                    if argument.children or captured is None:
                        reject(
                            argument.token,
                            "callback.signature",
                            "expected a bound zero-argument callback",
                        )
                    effects = captured
                child.zero_callbacks[name] = effects
                continue
            if parameter.kind in BORROWED_EFFECT_SIGNATURES:
                child.callback_bindings[name] = self.borrowed_effect_callback(
                    argument, BORROWED_EFFECT_SIGNATURES[parameter.kind]
                )
                continue
            if parameter.kind == CALLBACK_ARRAY:
                child.callback_arrays[name] = capture_callbacks(self, argument)
                continue
            if parameter.role == "callable":
                actual = self.atom(argument)
                callback = self.callback_bindings.get(
                    actual.text
                ) or self.definitions.get(actual.text)
                if (
                    callback is None
                    or callback.name.text == "main"
                    or len(callback.parameters) != 2
                ):
                    reject(
                        actual,
                        "signature",
                        "callback requires a named (real, qbit[]) helper",
                    )
                first, second = callback.parameters
                if (
                    first.role != "real"
                    or first.kind != "real"
                    or second.role != "inout"
                    or second.kind != "qbit[]"
                    or second.record is not None
                ):
                    reject(
                        actual,
                        "signature",
                        "callback requires a named (real, qbit[]) helper",
                    )
                child.callback_bindings[name] = callback
                continue
            if parameter.kind == "IndexedPauli[]":
                child.indexed_pauli[name] = bind_indexed_pauli(self, argument)
                continue
            if parameter.kind == "PauliTerm[]":
                terms = argument
                if not terms.children:
                    symbol = terms.token.text
                    terms = (
                        self.pauli_bindings.get(symbol, terms)
                        if symbol in self.parameters
                        else self.constants.get(symbol, terms)
                    )
                if (
                    terms.token.text != "PauliTerm[]"
                    or not 1 <= len(terms.children) <= 1024
                ):
                    reject(
                        argument.token,
                        "pauli",
                        "helper requires a static PauliTerm array",
                    )
                child.pauli_bindings[name] = terms
                continue
            if parameter.role == "integer":
                if parameter.kind in {"int[]", "Pauli[]"}:
                    if (
                        parameter.kind == "Pauli[]"
                        and argument.token.text == "$pauli_axes"
                    ):
                        values_int = tuple(
                            "IXYZ".index(item.token.text) for item in argument.children
                        )
                    else:
                        values_int = self.integer_array(argument)
                    if parameter.kind == "Pauli[]" and any(
                        item not in range(4) for item in values_int
                    ):
                        reject(
                            argument.token,
                            "pauli",
                            "Pauli array contains an invalid axis",
                        )
                    child.integer_arrays[name] = checked_integer_array(
                        values_int, argument.token, parameter.width
                    )
                else:
                    child.classical_integers[name] = checked_integer(
                        self.static_integer(argument), argument.token
                    )
                continue
            if parameter.role == "real":
                if parameter.kind == "real[][]":
                    matrix = bound_matrix(self, argument)
                    if not matrix or any(len(row) != parameter.width for row in matrix):
                        reject(
                            argument.token,
                            "binding",
                            "real matrix does not match the formal row width",
                        )
                    child.classical_matrices[name] = matrix
                elif parameter.kind == "real[]":
                    values = self.real_array(argument)
                    if not values or (
                        parameter.width is not None and len(values) != parameter.width
                    ):
                        reject(
                            argument.token,
                            "binding",
                            "helper real array length disagrees",
                        )
                    child.classical_arrays[name] = values
                else:
                    child.classical_values[name] = self.angle(argument, token)
                continue
            field_port = (
                self.record_field(argument) if argument.token.text == "." else None
            )
            actual_port = (
                field_port[0]
                if field_port is not None
                else (
                    self.parameters.get(argument.token.text)
                    if not argument.children
                    else None
                )
            )
            signed_actual = (
                actual_port is not None
                and actual_port.kind == "qnum_signed"
                or not argument.children
                and argument.token.text in self.signed
            )
            if parameter.kind == "qnum_signed" and not signed_actual:
                reject(
                    argument.token,
                    "type",
                    "signed helper requires a signed integer actual",
                )
            if (
                actual_port is not None
                and actual_port.kind == "qphase_signed"
                and (
                    parameter.kind != "qbit[]"
                    or parameter.role not in {"const", "inout"}
                )
            ):
                reject(
                    argument.token,
                    "precision",
                    "signed phase storage requires a borrowed bit view",
                )
            borrowed_counter = (
                name in modular_phase_ports
                and actual_port is not None
                and actual_port.kind == "qphase"
                and actual_port.role != "const"
                and parameter.role == "inout"
                and parameter.kind == "qnum"
            )
            if borrowed_counter or (
                actual_port is not None
                and actual_port.kind == "qphase"
                and parameter.kind == "qnum"
                and parameter.width is None
                and parameter.role in {"const", "inout"}
            ):
                parameter = replace(parameter, kind="qphase")
                child.parameters[name] = parameter
            if (
                actual_port is not None
                and actual_port.kind == "qphase"
                and (
                    parameter.kind not in {"qphase", "qbit[]"}
                    or (actual_port.role == "const" and parameter.role != "const")
                )
            ):
                reject(
                    argument.token,
                    "precision",
                    "fixed-point helper ports require a checked bit view or matching type",
                )
            if parameter.kind == "qphase" and (
                actual_port is None or actual_port.kind != "qphase"
            ):
                reject(
                    argument.token,
                    "precision",
                    "fixed-point helper ports require matching unsigned fractional storage",
                )
            if parameter.role == "output" and (
                actual_port is None or actual_port.role != "output"
            ):
                reject(
                    argument.token,
                    "ownership",
                    "bound helper Output requires a whole caller Output",
                )
            if (
                parameter.role == "output"
                and self.widths.get(argument.token.text) is None
            ):
                if any(
                    previous.text == argument.token.text
                    for previous in deferred.values()
                ):
                    reject(
                        argument.token,
                        "alias",
                        "writable helper arguments must be disjoint",
                    )
                deferred[name] = argument.token
                continue
            view = (
                record_view(self, parameter, argument)
                if parameter.record is not None
                else (
                    self.array_view(argument, allow_phase=True, allow_signed=True)
                    if parameter.kind in {"qbit[]", "qnum", "qnum_signed", "qphase"}
                    else (self.gate_target(argument, allow_numeric=True),)
                )
            )
            if parameter.length is not None and (
                actual_port is None or actual_port.length != parameter.length
            ):
                reject(argument.token, "width", "helper numeric array shape disagrees")
            if parameter.width is not None and len(view) != parameter.width:
                reject(argument.token, "width", "helper quantum array length disagrees")
            if len(set(view)) != len(view) or actuals.intersection(view):
                reject(
                    argument.token,
                    "alias",
                    "writable helper arguments must be disjoint",
                )
            actuals.update(view)
            child.widths[name] = len(view)
            if (
                actual_port is not None
                and actual_port.kind == "qphase_signed"
                and parameter.role == "inout"
            ):
                signed_bit_exports[argument.token.text] = len(view)
            if parameter.kind in {"qnum", "qnum_signed"} and signed_actual:
                child.signed.add(name)
            value = view[0][0]
            if view == tuple((value, index) for index in range(register_width(value))):
                child.registers[name] = value
            else:
                child.views[name] = view
            if borrowed_counter:
                last = next(
                    (
                        op
                        for op in reversed(self.operations)
                        if value in op.operands[1:]
                    ),
                    None,
                )
                if name not in child.registers or not (
                    last is not None
                    and last.name == "qreg.allocate"
                    or last is None
                    and value in self.borrowed_zero_phases
                ):
                    reject(
                        argument.token,
                        "qpe",
                        "phase output must be initialized by allocate and unused",
                    )
                child.borrowed_zero_phases.add(value)
                phase_exports[name] = argument.token.text
            if parameter.role == "output":
                if name not in child.registers:
                    reject(
                        argument.token,
                        "ownership",
                        "bound helper Output cannot be a partial view",
                    )
                outputs.add(value)
            if parameter.kind == "qphase" and parameter.role == "const":
                phase_borrows.add(value)
            elif parameter.role == "const":
                readonly.update(view)
        if not lower_narrow_pauli_sequence(child, syntax.statements):
            snapshot = snapshot_start(syntax.statements)
            for index, statement in enumerate(syntax.statements):
                if index + 3 == len(syntax.statements) and (
                    lower_computed_rotation(child, syntax.statements[index:])
                    or lower_fixed_payoff(child, syntax.statements[index:])
                    or lower_copied_selector(
                        child, syntax.statements[index:], str(index)
                    )
                ):
                    break
                if index == snapshot:
                    lower_snapshot(child, syntax.statements, index)
                    break
                child.statement(statement, str(index))
        if any(port.name.text in child.suspended for port in syntax.parameters):
            reject(
                token,
                "binding",
                "bound helper must restore every formal port before return",
            )
        materialized: dict[Value, Value] = {}
        for name, actual in deferred.items():
            child_output = child.registers.get(name)
            if child_output is None:
                reject(actual, "ownership", "bound helper must initialize each Output")
            parameter = child.parameters[name]
            if parameter.kind == "qphase":
                if self.parameters[actual.text].kind != "qnum":
                    reject(
                        actual,
                        "precision",
                        "inferred fractional Output requires a numeric caller port",
                    )
                self.parameters[actual.text] = replace(
                    self.parameters[actual.text], kind="qphase"
                )
            output = self.register(
                actual,
                register_width(child_output),
                allow_qubit_array=True,
                allow_fractional_output=parameter.kind == "qphase",
            )
            materialized[child_output] = output
            outputs.add(child_output)
        readonly_owners = {owner for owner, _ in readonly}
        initialized_outputs = set()
        for op in child.operations:
            operands = op.operands[1:]
            if op.name in {"qreg.allocate", "qreg.prepare_amplitudes"}:
                initialized_outputs.add(operands[0])
            if readonly_owners.intersection(operands) and not preserves_basis_labels(
                op, readonly
            ):
                reject(
                    token, "ownership", "bound helper cannot write a Const quantum port"
                )
            if phase_borrows.intersection(operands):
                if op.name == "qreg.control_phase" or (
                    op.name == "qreg.pair_gate"
                    and op.attributes.get("gate") == "cx"
                    and operands[0] in phase_borrows
                    and operands[1] not in phase_borrows
                ):
                    continue
                reject(
                    token,
                    "ownership",
                    "bound helper cannot write a Const fixed-point port",
                )
        if not outputs <= initialized_outputs:
            reject(token, "ownership", "bound helper must initialize each Output")
        if any(op.name == "qreg.hidden" for op in child.operations):
            reject(token, "free", "bound helper must free each allocated local qbit")
        for name, actual_name in phase_exports.items():
            if name in child.fractional_bits:
                self.fractional_bits[actual_name] = child.fractional_bits[name]
        self.fractional_bits.update(signed_bit_exports)
        self.helper_expansions = child.helper_expansions
        self.loop_iterations = child.loop_iterations
        for op in child.operations:
            operands = tuple(
                materialized.get(value, value) for value in op.operands[1:]
            )
            if op.name == "qreg.local":
                reject(
                    token,
                    "scope",
                    "bound helper needs a checked lowering for computed local storage",
                )
            self.effect_op(op.name, token, operands, **op.attributes)

    def call(
        self,
        token: Token,
        arguments: tuple[Expression, ...],
        *,
        const_only: bool = False,
        control_bit: Value | None = None,
        condition_inputs: tuple[Value, ...] = (),
    ) -> None:
        callback = self.callback_bindings.get(token.text)
        if callback is not None:
            if const_only or control_bit is not None:
                reject(token, "scope", "callbacks require ordinary static calls")
            start = len(self.operations)
            self.call_bound_helper(callback, token, arguments)
            if self.parameters[token.text].kind in BORROWED_EFFECT_SIGNATURES and any(
                op.name not in PURE_EFFECTS for op in self.operations[start:]
            ):
                reject(
                    token,
                    "callback.escape",
                    "borrowed callback requires pure unitary effects",
                )
            return
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
            if (
                parameter.record is not None
                and self.parameters.get(actual_name.text) is not None
                and self.parameters[actual_name.text].record != parameter.record
            ):
                reject(
                    actual_name,
                    "signature",
                    "record helper requires the matching record type",
                )
            actual = self.register(
                actual_name,
                None if borrowed else formal.type.parameters[0],
                allow_qubit_array=parameter.record is not None
                or parameter.kind == "qbit[]",
                allocate_local=not borrowed and self.scope == "main",
            )
            actual_kind = self.parameters[actual_name.text].kind
            raw_array_output = (
                not borrowed
                and parameter.kind == "qbit[]"
                and actual_kind == "qnum"
                and parameter.record is None
                and self.parameters[actual_name.text].role == "local"
            )
            if (
                actual_name.text in self.signed
                or actual_kind != parameter.kind
                and not raw_array_output
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

    def real_phase(
        self,
        token: Token,
        expression: Expression,
        coefficient: Expression,
        *,
        condition: Expression | None = None,
        disjoint_condition_target: tuple[Value, int] | None = None,
        conjugate_xor: bool = False,
        phase_sign: bool = False,
    ) -> None:
        saw_signed = False

        def signed_word(node: Expression) -> bool:
            nonlocal saw_signed
            if node.token.text == "." and node.children[1].token.text not in {
                "size",
                "len",
            }:
                selected = self.record_field(node)
                signed = selected is not None and selected[0].kind == "qnum_signed"
            else:
                signed = not node.children and node.token.text in self.signed
            saw_signed |= signed
            return signed

        def quantum_word(node: Expression) -> tuple[tuple[Value, int], ...] | None:
            if node.token.text == "$predicate_word":
                return self.array_view(node.children[0])
            cell = self.record_bit(node)
            if cell is not None:
                return (cell,)
            source = node.children[0] if node.token.text == "[" else node
            if source.token.text == ".":
                if source is node and source.children[1].token.text in {"size", "len"}:
                    return None
                resolved = self.record_field(source)
                assert resolved is not None
                field, view = resolved
                if node.token.text == "[":
                    if field.kind == "qbit[][]":
                        reject(
                            node.token,
                            "field",
                            "phase bit fields require row and column indices",
                        )
                    if field.kind not in {"qbit[]", "qnum[]"}:
                        reject(node.token, "field", "indexing requires an array field")
                    index = self.static_integer(node.children[1])
                    if not 0 <= index < (field.length or len(view)):
                        reject(node.token, "width", "field index is out of bounds")
                    if field.kind == "qnum[]":
                        assert field.width is not None
                        return view[index * field.width : (index + 1) * field.width]
                    return (view[index],)
                if field.kind not in {"qbit", "qnum", "qnum_signed"}:
                    reject(node.token, "field", "phase arrays require an element index")
                return view
            if source.children:
                return None
            parameter = self.parameters.get(source.token.text)
            if parameter is None or parameter.kind not in {
                "qbit",
                "qbit[]",
                "qnum",
                "qphase",
            }:
                return None
            if parameter.kind == "qbit[]" and node.token.text != "[":
                reject(
                    node.token,
                    "phase",
                    "phase arithmetic requires indexed array elements",
                )
            view = self.array_view(
                source,
                allow_signed=signed_word(node),
                allow_phase=parameter.kind == "qphase",
            )
            if node.token.text == "[":
                index = self.static_integer(node.children[1])
                if not 0 <= index < len(view):
                    reject(node.token, "width", "phase bit index is out of bounds")
                return (view[index],)
            return view

        def phase_scale(node: Expression) -> int:
            parameter = (
                self.parameters.get(node.token.text) if not node.children else None
            )
            return (
                (self.widths[node.token.text] or 0)
                if parameter is not None and parameter.kind == "qphase"
                else 0
            )

        factors: tuple[Cube, ...] = ()
        factor_values: tuple[Value, ...] = ()
        try:
            polynomial, locations = expand_phase(
                expression,
                quantum_word,
                self.classical_number,
                signed_word=signed_word,
                fraction_bits=phase_scale,
                condition=condition,
                disjoint_condition_target=disjoint_condition_target,
            )
        except HybridVerificationError as error:
            if (
                conjugate_xor
                and not saw_signed
                and all(d.code == "qmod.budget.phase" for d in error.diagnostics)
            ):
                assert disjoint_condition_target is not None
                owner, wire = disjoint_condition_target
                terms, reads = expand_phase(
                    Expression(Token("1", token.location)),
                    quantum_word,
                    self.classical_number,
                    condition=condition,
                    disjoint_condition_target=disjoint_condition_target,
                )
                for value, bit in reads:
                    self.effect_op(
                        "qreg.control_phase", token, (value,), indices=(bit,), angle=0.0
                    )
                for mask, weight in sorted(terms.items()):
                    if int(weight) % 2 == 0:
                        continue
                    if mask:
                        controls = tuple(
                            location
                            for i, location in enumerate(reads)
                            if mask & (1 << i)
                        )
                        self.emit_indexed_x(
                            token, (*controls, disjoint_condition_target)
                        )
                    else:
                        self.effect_op(
                            "qreg.gate", token, (owner,), gate="x", index=wire
                        )
                return
            if (
                condition is not None
                and disjoint_condition_target is None
                and not saw_signed
                and all(
                    diagnostic.code == "qmod.budget.phase"
                    for diagnostic in error.diagnostics
                )
            ):
                lower_decision_phase(
                    self, token, expression, coefficient, condition, quantum_word
                )
                return
            # Preserve the established small-polynomial path and its resources.
            # Only expansion-budget refusals admit this exact alternative.
            if condition is not None or not all(
                diagnostic.code == "qmod.budget.phase"
                for diagnostic in error.diagnostics
            ):
                raise
            factored, factors = preserve_factors(
                expression,
                lambda node: None if signed_word(node) else quantum_word(node),
            )
            if not factors:
                raise
            factor_values = tuple(
                Value(ValueId(self.scope, self.counter + i), register_type(1))
                for i in range(len(factors))
            )

            def factored_word(node: Expression) -> tuple[tuple[Value, int], ...] | None:
                if node.token.text.startswith("$phase_factor"):
                    return ((factor_values[int(node.token.text[13:])], 0),)
                return quantum_word(node)

            polynomial, locations = expand_phase(
                factored, factored_word, self.classical_number, signed_word=signed_word
            )
            physical_reads = {
                location for factor in factors for location, _ in factor
            } | {location for location in locations if location[0] not in factor_values}
            if len(physical_reads) > MAX_PHASE_INPUT_BITS:
                reject(token, "budget.phase", "phase exceeds 24 quantum input bits")
            self.counter += len(factors)
        if conjugate_xor and saw_signed:
            assert disjoint_condition_target is not None
            emit_signed_predicate_xor(
                self, token, locations, polynomial, disjoint_condition_target
            )
            return
        angle = (
            math.pi
            if not coefficient.children
            and coefficient.token.text == "pi"
            and "pi" not in self.parameters
            else self.classical_number(coefficient)
        )
        if conjugate_xor:
            assert disjoint_condition_target is not None
            owner, wire = disjoint_condition_target
            self.effect_op("qreg.gate", token, (owner,), gate="h", index=wire)
        for factor, value in zip(factors, factor_values, strict=True):
            self.phase_factor(token, factor, value)
        # Check reads even when algebraic cancellation removes their phase terms.
        for value, wire in locations:
            self.effect_op(
                "qreg.control_phase", token, (value,), indices=(wire,), angle=0.0
            )
        for mask, coefficient_value in sorted(polynomial.items()):
            if phase_sign:
                # The folded bit contributes only (-1) to the selected basis states.
                # Even terms of this Boolean Möbius expansion are exact identities.
                assert coefficient_value == int(coefficient_value) and angle == math.pi
                if not int(coefficient_value) % 2:
                    continue
                coefficient_value = 1
            radians = angle * coefficient_value
            if not math.isfinite(radians) or abs(radians) > 2**53:
                reject(token, "phase", "phase angle exceeds finite ±2**53 range")
            if not mask:
                self.effect_op("qreg.phase", token, (), angle=radians)
                continue
            selected = tuple(
                location for bit, location in enumerate(locations) if mask & (1 << bit)
            )
            self.effect_op(
                "qreg.control_phase",
                token,
                tuple(register for register, _ in selected),
                indices=tuple(wire for _, wire in selected),
                angle=radians,
            )
        for value in reversed(factor_values):
            self.effect_op("qreg.uncompute", token, (value,))
            self.effect_op("qreg.release", token, (value,))

        if conjugate_xor:
            assert disjoint_condition_target is not None
            owner, wire = disjoint_condition_target
            self.effect_op("qreg.gate", token, (owner,), gate="h", index=wire)

    def phase_factor(self, token: Token, factor: Cube, value: Value) -> None:
        owners = tuple(dict.fromkeys(owner for (owner, _), _ in factor))
        widths = {f"owner{i}": register_width(owner) for i, owner in enumerate(owners)}

        def node(text: str, *children: Expression) -> Expression:
            return Expression(replace(token, text=text), children)

        predicates = []
        for i, owner in enumerate(owners):
            mask = sum(1 << bit for (source, bit), _ in factor if source == owner)
            expected = sum(
                1 << bit
                for (source, bit), positive in factor
                if source == owner and positive
            )
            predicates.append(
                node(
                    "==",
                    node("&", node(f"owner{i}"), node(str(mask))),
                    node(str(expected)),
                )
            )
        predicate = predicates[0]
        for other in predicates[1:]:
            predicate = node("&", predicate, other)
        expression = _Expression(
            predicate, widths, f"{self.scope}.factor{value.id.index}", set()
        )
        graph = expression.function()
        self.functions.append(graph)
        self.append_op(
            token,
            Operation(
                "qreg.local",
                results=(value,),
                attributes={"name": graph.name},
                location=token.location,
            ),
        )
        self.effect_op("qreg.compute", token, (value, *owners), callee=graph.name)

    def bind_registers(
        self,
        token: Token,
        source: Expression,
        destination: Expression,
        *,
        publish_output: bool = False,
    ) -> None:
        sources = source.children if source.token.text == "{" else (source,)
        destinations = (
            destination.children if destination.token.text == "{" else (destination,)
        )
        source_names = tuple(self.atom(item).text for item in sources)
        destination_names = tuple(self.atom(item).text for item in destinations)
        if len(set(source_names + destination_names)) != len(
            source_names + destination_names
        ):
            reject(token, "binding", "binding names must be distinct")
        locations = tuple(
            location for item in sources for location in self.array_view(item)
        )
        if len(set(locations)) != len(locations):
            reject(token, "binding", "binding source locations must be disjoint")
        offset = 0
        assignments: list[tuple[str, tuple[tuple[Value, int], ...]]] = []
        for item in destinations:
            name = self.atom(item).text
            parameter = self.parameters.get(name)
            if parameter is None or parameter.role not in {"local", "output", "inout"}:
                reject(
                    item.token,
                    "binding",
                    "binding destination must be a declared local or output",
                )
            if name in self.views and name not in self.suspended:
                reject(
                    item.token, "binding", "binding destination already owns storage"
                )
            width = self.widths[name]
            if width is None:
                if len(destinations) != 1:
                    reject(
                        item.token,
                        "binding",
                        "split destinations require explicit widths",
                    )
                width = len(locations)
            view = locations[offset : offset + width]
            offset += width
            if len(view) != width:
                reject(
                    item.token,
                    "binding",
                    "binding widths must cover the source exactly",
                )
            if name in self.registers:
                value = self.registers[name]
                expected = tuple(
                    (value, index) for index in range(register_width(value))
                )
                if name not in self.suspended or view != expected:
                    reject(
                        item.token,
                        "binding",
                        "restored storage must match the original wire order",
                    )
            elif name in self.views:
                if name not in self.suspended or view != self.views[name]:
                    reject(
                        item.token,
                        "binding",
                        "restored storage must match the original wire order",
                    )
            elif parameter.role != "local":
                if parameter.role != "output":
                    reject(
                        item.token,
                        "binding",
                        "borrowed storage must be restored, not rebound",
                    )
                if not publish_output or len(destinations) != 1:
                    reject(
                        item.token,
                        "binding",
                        "fresh output binding requires an allocated storage owner",
                    )
                publish_bound_array(self, item.token, view)
            assignments.append((name, view))
        if offset != len(locations):
            reject(token, "binding", "binding widths must cover the source exactly")
        # Identity effects let the shared verifier check source initialization.
        checked: set[Value] = set()
        for value, wire in locations:
            if value not in checked:
                self.effect_op("qreg.gate", token, (value,), gate="i", index=wire)
                checked.add(value)
        self.suspended.update(source_names)
        for name, view in assignments:
            self.widths[name] = len(view)
            self.suspended.discard(name)
            if name in self.registers:
                self.views.pop(name, None)
            else:
                self.views[name] = view

    def increment(
        self,
        token: Token,
        target: Expression,
        amount: Expression,
        *,
        controls: tuple[tuple[Value, int], ...] = (),
    ) -> None:
        field = self.record_field(target)
        name = target.token.text if field is not None else self.atom(target).text
        parameter = field[0] if field is not None else self.parameters.get(name)
        if parameter is None or parameter.kind != "qnum":
            reject(target.token, "type", "in-place increment requires an integer")
        if name in self.fractional_bits:
            reject(target.token, "type", "constant increment requires an integer")
        negative = amount.token.text == "-" and len(amount.children) == 1
        magnitude = amount.children[0] if negative else amount
        if magnitude.children or not magnitude.token.text.isdecimal():
            if field is not None or controls or name in self.signed:
                reject(
                    target.token, "type", "signed increment requires a constant offset"
                )
            lower_affine_update(self, token, target, amount)
            return
        if int(magnitude.token.text) > 256:
            reject(
                amount.token,
                "arithmetic",
                "constant increment requires a literal offset in -256..256",
            )
        view = self.array_view(target, allow_signed=True)
        modulus = 1 << len(view)
        offset = ((-1 if negative else 1) * int(magnitude.token.text)) % modulus
        if offset > modulus // 2:
            offset -= modulus
        if offset == 0:
            value, bit = view[0]
            self.effect_op("qreg.gate", token, (value,), gate="i", index=bit)
        for _ in range(abs(offset)):
            order = range(len(view)) if offset < 0 else reversed(range(len(view)))
            for bit in order:
                if bit or controls:
                    self.emit_indexed_x(token, (*controls, *view[:bit], view[bit]))
                else:
                    value, wire = view[0]
                    self.effect_op("qreg.gate", token, (value,), gate="x", index=wire)

    def emit_indexed_x(
        self, token: Token, locations: tuple[tuple[Value, int], ...]
    ) -> None:
        if len(locations) < 2 or len(set(locations)) != len(locations):
            reject(
                token, "control", "controls and target must be distinct qubit locations"
            )
        self.effect_op(
            "qreg.control_x",
            token,
            tuple(value for value, _ in locations),
            indices=tuple(wire for _, wire in locations),
        )

    def coherent_helper_control(self, statement: Statement, index: str) -> bool:
        condition = statement.arguments[0]
        if condition.token.text in {"==", "!="} and len(condition.children) == 2:
            source, expected = condition.children
            if expected.children or expected.token.text not in {"0", "1"}:
                return False
            positive = int(expected.token.text) ^ (condition.token.text == "!=")
        elif condition.token.text == "[" or (
            not condition.children
            and self.widths.get(condition.token.text) == 1
            and self.parameters[condition.token.text].kind in {"qbit", "qnum"}
        ):
            source, positive = condition, 1
        else:
            return False
        if source.token.text == "[":
            control = self.gate_target(source)
        elif not source.children:
            parameter = self.parameters.get(source.token.text)
            if (
                parameter is None
                or parameter.kind not in {"qbit", "qnum"}
                or self.widths[source.token.text] != 1
            ):
                return False
            control = self.array_view(source)[0]
        else:
            return False
        body = statement.control_body
        assert body is not None
        branches: tuple[tuple[int, tuple[Statement, ...]], ...] = ((positive, body),)
        if statement.control_else is not None:
            branches += ((1 - branches[0][0], statement.control_else),)

        # Nested control blocks enter this path when they contain a helper composition.
        def contains_composition(children: tuple[Statement, ...]) -> bool:
            return any(
                child.inverse_body is not None
                or child.repeat_index is not None
                or child.within is not None
                or child.token.text in {"+=", "unitary"}
                or child.token.text in self.definitions
                or (
                    child.control_body is not None
                    and contains_composition(child.control_body)
                )
                or (
                    child.control_else is not None
                    and contains_composition(child.control_else)
                )
                for child in children
            )

        if not any(contains_composition(children) for _, children in branches):
            return False
        for value, children in branches:
            start, incoming = len(self.operations), self.effect
            for ordinal, child in enumerate(children):
                if (
                    child.token.text == "IDENTITY"
                    and len(child.arguments) == 1
                    and control in self.array_view(child.arguments[0])
                ):
                    reject(child.token, "alias", "control and target must be disjoint")
                self.statement(child, f"{index}.control{value}.{ordinal}")
            expanded = self.operations[start:]
            del self.operations[start:]
            self.effect = incoming
            if not value:
                self.effect_op(
                    "qreg.gate",
                    statement.token,
                    (control[0],),
                    gate="x",
                    index=control[1],
                )
            for (name, operands, attrs), run in groupby(
                expanded, key=lambda op: (op.name, op.operands[1:], dict(op.attributes))
            ):
                if (
                    name == "qreg.unitary"
                    and "control_index" not in attrs
                    and "power" not in attrs
                    and control[0] not in operands
                ):
                    power = sum(1 for _ in run)
                    for start in range(0, power, 128):
                        self.effect_op(
                            name,
                            statement.token,
                            (*operands, control[0]),
                            **attrs,
                            control_index=control[1],
                            power=min(128, power - start),
                        )
                    continue
                for effect in run:
                    if (
                        "control_state" in attrs
                        or name == "qreg.inplace_prepare_amplitudes"
                        or name == "qreg.gate"
                        and attrs.get("gate") in {"phase", "z", "s"}
                    ):
                        emit_selected(self, statement.token, (control,), 1, (effect,))
                        continue
                    if name == "qreg.control_x" and "indices" in attrs:
                        locations = tuple(zip(operands, attrs["indices"], strict=True))
                    elif name == "qreg.gate" and attrs["gate"] in {"i", "x"}:
                        locations = ((operands[0], attrs["index"]),)
                        if attrs["gate"] == "i":
                            if control in locations:
                                reject(
                                    statement.token,
                                    "alias",
                                    "control and target must be disjoint",
                                )
                            self.effect_op(name, statement.token, operands, **attrs)
                            continue
                    elif name == "qreg.pauli_rotation":
                        selectors = (
                            (control, (operands[1], attrs["control_index"]))
                            if "control_index" in attrs
                            else (control,)
                        )
                        emit_controlled_pauli(
                            self,
                            statement.token,
                            operands[0],
                            attrs["support"],
                            attrs["angle"],
                            selectors,
                        )
                        continue
                    elif name == "qreg.phase" or (
                        name == "qreg.control_phase" and "indices" in attrs
                    ):
                        phase_inputs = tuple(
                            zip(operands, attrs.get("indices", ()), strict=True)
                        )
                        if control in phase_inputs:
                            reject(
                                statement.token,
                                "alias",
                                "control and target must be disjoint",
                            )
                        selected = (control, *phase_inputs)
                        self.effect_op(
                            "qreg.control_phase",
                            statement.token,
                            tuple(value for value, _ in selected),
                            indices=tuple(wire for _, wire in selected),
                            angle=attrs["angle"],
                        )
                        continue
                    elif name == "qreg.gate" and attrs["gate"] in {
                        "rx",
                        "ry",
                        "rz",
                        "h",
                    }:
                        destination = operands[0]
                        wire = attrs["index"]
                        if control == (destination, wire):
                            reject(
                                statement.token,
                                "alias",
                                "control and target must be disjoint",
                            )
                        if attrs["gate"] == "h":
                            self.effect_op(
                                "qreg.control_phase",
                                statement.token,
                                (control[0], destination),
                                indices=(control[1], wire),
                                angle=math.pi,
                            )
                            axis, angle = "Y", math.pi / 4
                        else:
                            axis, angle = attrs["gate"][-1].upper(), attrs["angle"] / 2
                            if attrs.get("inverse"):
                                angle = -angle
                        support = ((wire, axis),)
                        if control[0] == destination:
                            emit_controlled_pauli(
                                self,
                                statement.token,
                                destination,
                                support,
                                angle,
                                (control,),
                            )
                        else:
                            self.pauli_exponential(
                                statement.token,
                                destination,
                                support,
                                angle,
                                control=control,
                            )
                        continue
                    elif name == "qreg.pair_gate" and attrs["gate"] in {"cx", "swap"}:
                        left, right = zip(
                            operands, attrs.get("indices", (0, 0)), strict=True
                        )
                        if control in (left, right):
                            reject(
                                statement.token,
                                "alias",
                                f"controlled {attrs['gate'].upper()} requires distinct locations",
                            )
                        if attrs["gate"] == "swap":
                            self.emit_indexed_x(statement.token, (right, left))
                        self.emit_indexed_x(statement.token, (control, left, right))
                        if attrs["gate"] == "swap":
                            self.emit_indexed_x(statement.token, (right, left))
                        continue
                    else:
                        reject(
                            statement.token,
                            "control",
                            "controlled helper requires reversible X, indexed controlled-X, SWAP or a disjoint unitary",
                        )
                    self.emit_indexed_x(statement.token, (control, *locations))
            if not value:
                self.effect_op(
                    "qreg.gate",
                    statement.token,
                    (control[0],),
                    gate="x",
                    index=control[1],
                )
        return True

    def controlled_rz_branches(self, statement: Statement) -> bool:
        yes, no = statement.control_body, statement.control_else
        if yes is None or no is None or len(yes) != 1 or len(no) != 1:
            return False
        if any(
            action.token.text != "RZ" or len(action.arguments) != 2
            for action in (*yes, *no)
        ):
            return False
        condition = statement.arguments[0]
        negative = Expression(Token("~", condition.token.location), (condition,))
        for branch, guard in ((yes[0], condition), (no[0], negative)):
            angle, marker = branch.arguments
            location = self.gate_target(marker, allow_numeric=True)
            half = Expression(Token("0.5", marker.token.location))
            weighted = Expression(Token("-", marker.token.location), (marker, half))
            self.real_phase(
                branch.token,
                weighted,
                angle,
                condition=guard,
                disjoint_condition_target=location,
            )
        return True

    def scalar_control(self, statement: Statement, index: str) -> None:
        token = statement.token
        body = statement.control_body
        if len(statement.arguments) != 1 or body is None:
            reject(token, "control", "control requires one condition and one operation")
        if any(
            child.token.text == "RESET"
            for child in (*body, *(statement.control_else or ()))
        ):
            reject(
                token,
                "control",
                "RESET is nonunitary and cannot be coherently controlled",
            )
        condition = statement.arguments[0]
        if (
            len(body) == 1
            and body[0].token.text == "^="
            and len(body[0].arguments) == 2
            and statement.control_else is None
            and lower_fixed_xor(
                self,
                body[0].token,
                body[0].arguments[0],
                body[0].arguments[1],
                guard=condition,
            )
        ):
            return
        if lower_masked_word_xor(self, statement):
            return
        if lower_guarded_selector(self, statement, index):
            return
        if lower_callback_control(self, statement):
            return
        if controlled_evolution(self, statement, index):
            return
        if lower_fourier_control(self, statement):
            return
        if lower_guarded_field_increment(self, statement):
            return
        if self.controlled_rz_branches(statement):
            return
        if lower_constant_phase_guard(self, statement):
            return
        if lower_masked_selection(self, statement):
            return
        if self.coherent_helper_control(statement, index):
            return
        if (
            len(body) == 1
            and body[0].token.text == "unitary"
            and statement.control_else is None
            and condition.token.text == "=="
        ):
            selector, literal = condition.children
            controls = self.array_view(selector)
            lower_unitary(
                self,
                body[0].token,
                body[0].arguments,
                selection=(controls, self.static_integer(literal)),
            )
            return
        if (
            len(body) == 1
            and statement.control_else is None
            and condition.token.text == "=="
            and not condition.children[0].children
            and (self.widths.get(condition.children[0].token.text) or 0) > 1
            and (
                body[0].token.text in {"grover_diffuser", "inplace_prepare_state"}
                or is_reversible_scope(body[0])
                or (
                    (helper := self.definitions.get(body[0].token.text)) is not None
                    and _bound_helper(helper)
                )
            )
        ):
            start, incoming = len(self.operations), self.effect
            self.statement(body[0], f"{index}.select")
            effects = tuple(self.operations[start:])
            del self.operations[start:]
            self.effect = incoming
            if is_reversible_scope(body[0]):
                selector, literal = condition.children
                controls = self.array_view(selector)
                expected = self.static_integer(literal)
                if not 1 <= len(controls) <= 7 or not 0 <= expected < 1 << len(
                    controls
                ):
                    reject(
                        token, "control", "selected scope equality must fit its word"
                    )
                emit_selected(self, token, controls, expected, effects)
            else:
                lower_selected_effects(self, token, condition, effects)
            return
        if len(body) != 1:
            reject(token, "control", "control requires one condition and one operation")
        if statement.control_else is not None:
            reject(
                token, "control", "control else requires a supported reversible helper"
            )
        operation = body[0]
        if operation.control_body is not None:
            if lower_nested_indexed_control(self, condition, operation):
                return
            reject(token, "scope", "nested scalar control requires function scope")
        if lower_zero_reflection_control(self, condition, operation):
            return
        if lower_indexed_control(self, condition, operation):
            return
        if condition.token.text == ":":
            if operation.token.text != "X" or len(operation.arguments) != 1:
                reject(token, "control", "slice control requires one X target")
            controls = self.array_view(condition)
            target_location = self.gate_target(operation.arguments[0])
            self.emit_indexed_x(token, (*controls, target_location))
            return
        args = operation.arguments
        if (
            operation.token.text == "phase"
            and len(args) == 2
            and (
                args[0].children
                or args[1].token.text in self.classical_values
                or args[1].token.text in self.classical_integers
                or args[1].token.text == "["
                or any(
                    parameter.record is not None
                    for parameter in self.parameters.values()
                )
            )
        ):
            self.real_phase(operation.token, args[0], args[1], condition=condition)
            return
        flip_phase = (
            operation.token.text == "phase"
            and len(args) == 1
            and not args[0].children
            and args[0].token.text == "pi"
            and "pi" not in self.parameters
        )
        if flip_phase and condition.token.text == "[":
            register, wire = self.gate_target(condition)
            self.effect_op("qreg.gate", token, (register,), gate="z", index=wire)
            return
        parameter = self.parameters.get(condition.token.text)
        if (
            not condition.children
            and parameter is not None
            and parameter.kind == "qbit[]"
        ):
            if flip_phase:
                if parameter.name.text in self.views:
                    view = self.array_view(condition)
                    if len(view) > 8:
                        phase_target, phase_bit = view[-1]
                        self.effect_op(
                            "qreg.gate",
                            token,
                            (phase_target,),
                            gate="h",
                            index=phase_bit,
                        )
                        self.emit_indexed_x(token, view)
                        self.effect_op(
                            "qreg.gate",
                            token,
                            (phase_target,),
                            gate="h",
                            index=phase_bit,
                        )
                    else:
                        self.effect_op(
                            "qreg.control_phase",
                            token,
                            tuple(value for value, _ in view),
                            indices=tuple(bit for _, bit in view),
                            angle=math.pi,
                        )
                else:
                    control = self.register(parameter.name, allow_qubit_array=True)
                    self.effect_op("qreg.array_phase_flip", token, (control,))
                return
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
        targets: tuple[Value, ...] = ()
        angle = None
        if operation.token.text in {"H", "hadamard_transform"} and len(args) == 1:
            targets = (self.register(self.atom(args[0])),)
            opcode = "qreg.control_h"
        elif flip_phase:
            angle = math.pi
            opcode = "qreg.control_phase"
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
            if opcode == "qreg.control_phase" and len(args) == 2:
                self.real_phase(operation.token, args[0], args[1], condition=condition)
                return
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
        if lower_binding_scope(self, statement, index):
            return
        if not statement.within:
            reject(
                statement.token,
                "scope",
                "within requires at least one local computation",
            )
        if (
            len(statement.within) == 1
            and statement.within[0].token.text == "H"
            and len(statement.apply) == 1
        ):
            action = statement.apply[0]
            helper = self.definitions.get(action.token.text)
            if (
                helper is not None
                and _bound_helper(helper)
                and action.control_body is None
            ):
                self.statement(statement.within[0], index)
                self.call(action.token, action.arguments)
                self.statement(statement.within[0], index)
                return
        if is_reversible_scope(statement):
            lower_reversible_scope(self, statement, index)
            return
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

    def linear_pauli_rotations(
        self, token: Token, args: tuple[Expression, ...]
    ) -> None:
        if len(args) != 5:
            reject(
                token,
                "signature",
                "linear_pauli_rotations requires bases, slopes, offsets, control and target",
            )
        bases, slopes, offsets, control_arg, target_arg = args

        def bit_view(node: Expression) -> tuple[tuple[Value, int], ...]:
            if node.token.text == "[" and not node.children[0].children:
                return (self.gate_target(node),)
            return self.array_view(node)

        controls = bit_view(control_arg)
        targets = bit_view(target_arg)
        if len(set((*controls, *targets))) != len(controls) + len(targets):
            reject(
                token, "alias", "linear_pauli_rotations requires disjoint qubit arrays"
            )
        width = len(targets)
        if any(len(node.children) != width for node in (bases, slopes, offsets)):
            reject(
                token, "width", "bases, slopes and offsets must match the target width"
            )
        for (target, wire), basis, slope, offset in zip(
            targets, bases.children, slopes.children, offsets.children, strict=True
        ):
            if basis.token.text not in {"Y", "2"}:
                reject(basis.token, "pauli", "linear_pauli_rotations supports only Y")
            self.effect_op(
                "qreg.gate",
                token,
                (target,),
                gate="ry",
                index=wire,
                angle=self.classical_number(offset),
            )
            coefficient = self.classical_number(slope)
            for index, (control, control_wire) in enumerate(controls):
                self.effect_op(
                    "qreg.pair_gate",
                    token,
                    (control, target),
                    gate="cry",
                    indices=(control_wire, wire),
                    angle=math.ldexp(coefficient, index),
                )

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

    def suzuki_trotter(
        self,
        token: Token,
        args: tuple[Expression, ...],
        *,
        emit: (
            Callable[[Token, Expression, float, Expression, int, int], None] | None
        ) = None,
    ) -> None:
        if len(args) != 5:
            reject(
                token,
                "signature",
                f"{token.text} requires terms, time, order, repetitions and target",
            )
        terms, time_arg, order_arg, repetitions_arg, target_arg = args
        if token.text == "multi_suzuki_trotter":
            if terms.token.text != "[" or not 1 <= len(terms.children) <= 16:
                reject(
                    token, "evolution", "multi_suzuki_trotter requires 1..16 operators"
                )
            times = self.real_array(time_arg)
            if len(times) != len(terms.children):
                reject(
                    time_arg.token,
                    "evolution",
                    "operator and time arrays must have equal lengths",
                )
            if len(times) > 1:
                if any(
                    arg.children or arg.token.text != "1"
                    for arg in (order_arg, repetitions_arg)
                ):
                    reject(
                        token,
                        "evolution",
                        "multiple operators require order 1 and one repetition",
                    )
                for operator, operator_time in zip(terms.children, times, strict=True):
                    self.pauli_evolution(
                        token, operator, operator_time, target_arg, 1, 1
                    )
                return
            terms, time = terms.children[0], times[0]
        else:
            time = self.evolution_angle(time_arg, token)
        computed = repetitions_arg.token.text == "$ceiling_repetitions"
        if (
            order_arg.children
            or not order_arg.token.text.isdecimal()
            or not computed
            and (repetitions_arg.children or not repetitions_arg.token.text.isdecimal())
        ):
            reject(
                token,
                "evolution",
                "order and repetitions must be integer literals or a static ceiling",
            )
        order = int(order_arg.token.text)
        repetitions = (
            ceiling_repetitions(self, repetitions_arg.children[0])
            if computed
            else int(repetitions_arg.token.text)
        )
        if order not in {1, 2} or not 1 <= repetitions <= 256:
            reject(token, "evolution", "use order 1 or 2 and 1..256 repetitions")
        if emit is not None:
            emit(token, terms, time, target_arg, order, repetitions)
        else:
            self.pauli_evolution(
                token, terms, time, target_arg, order, repetitions, compact=computed
            )

    def exponentiate(self, token: Token, args: tuple[Expression, ...]) -> None:
        if len(args) != 3:
            reject(
                token, "signature", "exponentiate requires an operator, time and target"
            )
        operator, time_arg, target_arg = args
        # Classiq's Hamiltonian Simulation guide specifies a first-order single step.
        self.pauli_evolution(
            token, operator, self.evolution_angle(time_arg, token), target_arg, 1, 1
        )

    def pauli_evolution(
        self,
        token: Token,
        operator: Expression,
        time: float,
        target_arg: Expression,
        order: int,
        repetitions: int,
        *,
        compact: bool = False,
    ) -> None:
        register, terms = self.pauli_terms(token, operator, target_arg)
        step = time / repetitions
        if order == 2:
            step /= 2
        sequence = terms if order == 1 else [*terms, *reversed(terms)]
        if compact:
            angles = [(support, step * weight) for support, weight in sequence]
            coalesced = compact_commuting(token, angles, repetitions)
            if coalesced is not None:
                for support, angle in coalesced:
                    self.pauli_exponential(token, register, support, angle)
                return
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
        total = math.fsum(weights)
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
        view = self.array_view(target_arg)
        owners = tuple(dict.fromkeys(value for value, _ in view))
        if len(owners) != 1 or len(set(view)) != len(view):
            reject(
                target_arg.token,
                "alias",
                "Pauli view requires distinct locations within one storage owner",
            )
        register = owners[0]
        width = len(view)
        if not operator.children and operator.token.text != "[":
            bound = self.pauli_bindings.get(operator.token.text)
            if bound is not None:
                operator = bound
            elif operator.token.text not in self.constants:
                family = (
                    "PauliTerm[]" if token.text == "qaoa_penalty" else "SparsePauliOp"
                )
                reject(operator.token, "pauli", f"unknown {family} constant")
            else:
                operator = self.constants[operator.token.text]
            if operator.token.text not in {"SparsePauliOp", "PauliTerm[]"}:
                reject(
                    target_arg.token, "pauli", "operator constant must be a Pauli sum"
                )
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
        kind = (
            self.parameters[target_arg.token.text].kind
            if not target_arg.children
            else "qbit[]"
        )
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
            mapped_support = tuple((view[index][1], axis) for index, axis in support)
            prepared.append((mapped_support, self.evolution_angle(coefficient, token)))
        return register, prepared

    def qpe_flexible(self, statement: Statement) -> None:
        token = statement.token
        assert statement.power_callback is not None
        power, body = statement.power_callback
        if (
            power.text in self.parameters
            or power.text in self.constants
            or power.text in _INTRINSICS
            or power.text in self.loop_indices
        ):
            reject(power, "qpe", "power parameter must be unshadowed")
        if is_modular_power(statement):
            lower_modular_qpe(self, statement)
            return
        body = specialize_power_body(self, body, power)
        if (
            len(body) != 1
            or body[0].token.text != "suzuki_trotter"
            or len(body[0].arguments) != 5
        ):
            reject(token, "qpe", "power callback requires one suzuki_trotter call")
        phase_name = self.atom(statement.arguments[0])
        parameter = self.parameters.get(phase_name.text)
        if (
            parameter is None
            or parameter.kind not in {"qphase", "qphase_signed"}
            or parameter.width is None
        ):
            reject(phase_name, "qpe", "expected an all-fractional phase output")
        width = parameter.width
        if width > 8:
            reject(
                phase_name,
                "qpe",
                "phase estimation supports at most eight phase qubits",
            )
        phase = self.register(phase_name, allow_fractional_output=True)
        last_use = next(
            (op for op in reversed(self.operations) if phase in op.operands[1:]), None
        )
        if last_use is None or last_use.name != "qreg.allocate":
            reject(
                phase_name,
                "qpe",
                "phase output must be initialized by allocate and unused",
            )
        operator, time, order, repetitions, target = body[0].arguments
        if any(arg.children or arg.token.text != "1" for arg in (order, repetitions)):
            reject(
                token, "qpe", "power callback requires first order and one repetition"
            )
        state, terms = self.pauli_terms(body[0].token, operator, target)

        def coefficient(node: Expression, value: int) -> float:
            text = node.token.text
            if text == "*" and len(node.children) == 2:
                left, right = node.children
                return coefficient(left, value) * coefficient(right, value)
            if text == "-" and len(node.children) == 1:
                return -coefficient(node.children[0], value)
            if not node.children:
                if text == power.text:
                    return float(value)
                if text == "pi" and "pi" not in self.parameters:
                    return math.pi
            return _literal_angle(node, token)

        for index in range(width):
            self.effect_op("qreg.gate", token, (phase,), gate="h", index=index)
        for index in range(width):
            evolution = coefficient(time, 1 << index)
            for support, weight in terms:
                angle = evolution * weight
                if not math.isfinite(2 * angle):
                    reject(
                        time.token, "qpe", "controlled evolution angle must be finite"
                    )
                self.pauli_exponential(
                    token, state, support, angle, control=(phase, index)
                )
        self.effect_op("qreg.inverse_qft", token, (phase,))
        self.fractional_bits[phase_name.text] = width

    def pauli_exponential(
        self,
        token: Token,
        register: Value,
        support: tuple[tuple[int, str], ...],
        angle: float,
        *,
        control: tuple[Value, int] | None = None,
    ) -> None:
        if not math.isfinite(angle) or not math.isfinite(2 * angle):
            reject(
                token, "angle", "Pauli rotation angle and doubled angle must be finite"
            )
        values = (register,) if control is None else (register, control[0])
        attributes: dict[str, object] = {"support": support, "angle": angle}
        if control is not None:
            attributes["control_index"] = control[1]
        self.effect_op("qreg.pauli_rotation", token, values, **attributes)

    def prepare_exponential_state(
        self, token: Token, args: tuple[Expression, ...]
    ) -> None:
        if len(args) != 2:
            reject(
                token,
                "signature",
                "prepare_exponential_state requires a rate and qubit array",
            )
        rate = self.evolution_angle(args[0], token)
        target = self.atom(args[1])
        register = self.register(target, allow_qubit_array=True)
        if self.parameters[target.text].kind != "qbit[]":
            reject(target, "type", "prepare_exponential_state requires a qubit array")
        for index, angle in enumerate(
            exponential_angles(rate, register_width(register))
        ):
            self.effect_op(
                "qreg.gate", token, (register,), gate="ry", index=index, angle=angle
            )

    def prepare_state_vector(self, token: Token, args: tuple[Expression, ...]) -> None:
        if len(args) != 3:
            reject(
                token,
                "signature",
                f"{token.text} requires a vector, bound and output",
            )
        vector, bound_arg, output = args
        values = (
            self.classical_arrays.get(vector.token.text)
            if not vector.children
            else None
        )
        if values is None and vector.token.text != "[":
            constant = self.constants.get(vector.token.text)
            if constant is None or constant.token.text != "[":
                reject(vector.token, "amplitudes", "expected a declared real[] vector")
            vector = constant
        if not bound_arg.children and bound_arg.token.text in self.constants:
            constant = self.constants[bound_arg.token.text]
            if constant.token.text in {"[", "SparsePauliOp"}:
                reject(bound_arg.token, "amplitudes", "bound requires a scalar real")
            bound_arg = constant
        count = len(values) if values is not None else len(vector.children)
        if not 2 <= count <= MAX_PREPARED_AMPLITUDES or count & (count - 1):
            reject(
                vector.token,
                "amplitudes",
                "amplitude count must be a power of two in 2..256",
            )

        amplitudes = (
            values
            if values is not None
            else tuple(_preparation_value(item) for item in vector.children)
        )
        bound = _preparation_value(bound_arg)
        inplace = token.text == "inplace_prepare_amplitudes"
        if inplace and (
            not math.isclose(math.hypot(*amplitudes), 1.0, abs_tol=1e-12, rel_tol=0)
            or not math.isfinite(bound)
            or bound < 0
        ):
            reject(
                vector.token,
                "amplitudes",
                "inplace preparation requires a normalized finite real vector and a nonnegative finite bound",
            )
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
        if inplace and (output.children or output.token.text in self.views):
            locations = tuple(reversed(self.array_view(output, allow_signed=True)))
            if len(locations) != count.bit_length() - 1:
                reject(
                    output.token,
                    "width",
                    "amplitude vector must match the output width",
                )
            self.effect_op(
                "qreg.inplace_prepare_amplitudes",
                token,
                tuple(value for value, _ in locations),
                indices=tuple(index for _, index in locations),
                amplitudes=amplitudes,
                bound=bound,
            )
            return
        target = self.atom(output)
        width = count.bit_length() - 1
        destination = self.register(
            target, width, allow_qubit_array=True, allocate_local=not inplace
        )
        if register_width(destination) != width:
            reject(target, "width", "amplitude vector must match the output width")
        self.effect_op(
            "qreg.inplace_prepare_amplitudes" if inplace else "qreg.prepare_amplitudes",
            token,
            (destination,),
            amplitudes=amplitudes,
            bound=bound,
        )

    def xor_update(
        self, token: Token, destination: Expression, source: Expression
    ) -> None:
        if source.token.text == "==":
            tested, literal = source.children
            if (
                not tested.children
                and not literal.children
                and literal.token.text.isdecimal()
                and 6 <= (self.widths.get(tested.token.text) or 0) <= 8
                and self.parameters[tested.token.text].role != "const"
                and lower_indexed_control(
                    self,
                    source,
                    Statement(Token("X", token.location), (destination,)),
                )
            ):
                return
        if source.token.text in _STATIC_COMPARISONS or source.token.text == "and":
            register, bit = self.gate_target(destination, allow_numeric=True)
            pi = Expression(Token("pi", token.location))
            self.real_phase(
                token,
                destination,
                pi,
                condition=source,
                disjoint_condition_target=(register, bit),
                conjugate_xor=True,
            )
            return
        field = self.record_field(destination)
        if field is not None:
            parameter, view = field
            if parameter.kind == "qbit" and source.token.text == ".":
                control = self.gate_target(source)
                self.emit_indexed_x(token, (control, view[0]))
                return
            if parameter.kind not in {"qbit", "qnum"}:
                reject(
                    destination.token, "type", "XOR requires an unsigned scalar field"
                )
            mask = self.static_integer(source)
            if not 0 <= mask < 1 << len(view):
                reject(source.token, "width", "XOR mask must fit the target width")
            for i, (owner, bit) in enumerate(view):
                if mask & (1 << i):
                    self.effect_op("qreg.gate", token, (owner,), gate="x", index=bit)
            if not mask:
                owner, bit = view[0]
                self.effect_op("qreg.gate", token, (owner,), gate="i", index=bit)
            return
        target = self.atom(destination)
        if (
            target.text in self.views
            and self.parameters[target.text].kind == "qbit"
            and self.parameters[target.text].role == "inout"
            and not source.children
            and source.token.text in {"0", "1"}
        ):
            value, index = self.gate_target(destination)
            self.effect_op(
                "qreg.gate",
                token,
                (value,),
                gate="x" if source.token.text == "1" else "i",
                index=index,
            )
            return
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
        if statement.callback_index is not None:
            for effect in indexed_callback(self, statement):
                self.effect_op(
                    effect.name, token, effect.operands[1:], **effect.attributes
                )
            return
        if statement.amplification is not None:
            lower_amplification(self, token, args[0], statement.amplification)
            return
        if token.text == "RESET":
            if len(args) != 1:
                reject(token, "signature", "RESET requires one initialized bit")
            value, wire = self.gate_target(args[0])
            self.effect_op("qreg.reset", token, (value,), index=wire)
            return
        if token.text == "swap_test":
            lower_swap_test(self, token, args)
            return
        if token.text == "$projected_rz":
            phase_angle, marker, condition = args
            location = self.gate_target(marker, allow_numeric=True)

            def literal(text: str, *children: Expression) -> Expression:
                return Expression(Token(text, token.location), children)

            # XOR conjugation maps the RZ exponent to t - 1/2 + b(1 - 2t).
            weighted = literal("-", literal("1"), literal("*", literal("2"), marker))
            self.real_phase(
                token,
                weighted,
                phase_angle,
                condition=condition,
                disjoint_condition_target=location,
            )
            self.real_phase(token, literal("-", marker, literal("0.5")), phase_angle)
            return
        if token.text == "qsvt_inversion":
            lower_qsvt_inversion(self, token, args)
            return
        if token.text == "grover_diffuser":
            lower_diffuser(self, token, args)
            return
        if token.text == "multiswap":
            if len(args) != 2:
                reject(token, "signature", "multiswap requires two register views")
            left_view, right_view = (self.array_view(arg) for arg in args)
            if len(left_view) != len(right_view) or len(
                set((*left_view, *right_view))
            ) != len(left_view) + len(right_view):
                reject(token, "alias", "multiswap needs equally sized disjoint views")
            for (first, i), (second, j) in zip(left_view, right_view, strict=True):
                self.effect_op(
                    "qreg.pair_gate",
                    token,
                    (first, second),
                    gate="swap",
                    indices=(i, j),
                )
            return
        if token.text == "inplace_prepare_state":
            lower_inplace_preparation(self, token, args)
            return
        if token.text == "select_rotation":
            lower_select_rotation(self, token, args)
            return
        if token.text == "qaoa_penalty":
            lower_qaoa_penalty(self, token, args)
            return
        if token.text == "full_hea":
            lower_hea(self, token, args)
            return
        if token.text == "CPHASE":
            if len(args) != 3:
                reject(token, "signature", "CPHASE requires an angle and two qubits")
            locations = tuple(self.gate_target(arg) for arg in args[1:])
            if len(set(locations)) != 2:
                reject(token, "alias", "CPHASE requires distinct qubit locations")
            self.effect_op(
                "qreg.control_phase",
                token,
                tuple(value for value, _ in locations),
                indices=tuple(index for _, index in locations),
                angle=self.angle(args[0], token),
            )
            return
        if statement.map_callback is not None:
            variable, body = statement.map_callback
            name = variable.text
            if (
                name in self.parameters
                or name in _INTRINSICS
                or name in self.constants
                or name in self.definitions
                or name in self.loop_indices
            ):
                reject(variable, "name", "map callback cannot shadow a bound symbol")
            if any(
                child.declaration is not None
                or child.token.text not in _GATE_CALLS | {"RX", "RY", "RZ"}
                for child in body
            ):
                reject(token, "scope", "map callback admits only single-qubit gates")
            locations = self.array_view(args[0])
            self.parameters[name] = Parameter(variable, "qbit", 1, "inout")
            self.widths[name] = 1
            try:
                for ordinal, location in enumerate(locations):
                    self.views[name] = (location,)
                    for gate_index, child in enumerate(body):
                        self.statement(child, f"{index}.map{ordinal}.{gate_index}")
            finally:
                self.views.pop(name, None)
                self.parameters.pop(name)
                self.widths.pop(name)
            return
        if token.text == "->":
            if self.scope == "main" and publish_partition(
                self, token, args[0], args[1]
            ):
                return
            self.bind_registers(token, args[0], args[1], publish_output=True)
            return
        if token.text == "+=":
            self.increment(token, args[0], args[1])
            return
        if statement.matrix_row is not None:
            if any(variable.text in _INTRINSICS for variable in statement.matrix_row):
                reject(token, "name", "real matrix row names cannot shadow intrinsics")
            lower_matrix_iteration(self, statement, index, MAX_LOOP_ITERATIONS)
            return
        if token.text in self.zero_callbacks:
            if statement != Statement(token, ()) or args:
                reject(
                    token,
                    "callback.signature",
                    "zero-argument callbacks do not accept operands or nested scopes",
                )
            for op in self.zero_callbacks[token.text]:
                self.effect_op(op.name, token, op.operands[1:], **op.attributes)
            return
        if statement.static_if_body is not None:
            if any(
                item.declaration is not None
                or (
                    item.token.text not in _STATIC_GATES
                    and item.static_if_body is None
                    and item.matrix_row is None
                    and item.control_body is None
                    and item.token.text not in self.definitions
                )
                for item in statement.static_if_body
            ):
                reject(token, "scope", "static if bodies admit only gates")
            condition = args[0]
            left, right = (self.static_integer(item) for item in condition.children)
            if _STATIC_COMPARISONS[condition.token.text](left, right):
                for ordinal, child in enumerate(statement.static_if_body):
                    self.statement(child, f"{index}.if{ordinal}")
            return
        if statement.inverse_body is not None:
            body = statement.inverse_body
            allowed = _GATE_CALLS | {
                "PHASE",
                "IDENTITY",
                "RX",
                "RY",
                "RZ",
                "inplace_encode_on_bloch",
                "inplace_prepare_state",
                "inplace_prepare_amplitudes",
                "qft",
            }
            if not body or any(
                child.token.text not in allowed
                and child.token.text not in self.definitions
                and child.token.text not in self.callback_bindings
                and child.token.text not in self.zero_callbacks
                and child.inverse_body is None
                and child.control_body is None
                for child in body
            ):
                reject(
                    token,
                    "scope",
                    "invert requires static single-qubit gates or reversible helpers",
                )
            start, incoming = len(self.operations), self.effect
            for ordinal, child in enumerate(body):
                self.statement(child, f"{index}.invert{ordinal}")
            forward = self.operations[start:]
            del self.operations[start:]
            self.effect = incoming
            emit_inverse(self, token, forward)
        elif statement.qpe_body is not None:
            lower_qpe(self, statement, index)
        elif statement.power_callback is not None:
            self.qpe_flexible(statement)
        elif statement.repeat_index is not None:
            self.repeat(statement, index)
        elif statement.declaration is not None:
            parameter = statement.declaration
            name = parameter.name.text
            if parameter.fractional_bits:
                reject(
                    parameter.name,
                    "precision",
                    "fixed-point locals require a checked computed rotation",
                )
            if name in self.parameters or name in _INTRINSICS or name in self.constants:
                reject(
                    parameter.name,
                    "name",
                    "local names cannot shadow parameters, locals or intrinsics",
                )
            width = parameter.width
            if parameter.width_expression is not None:
                width = self.static_integer(parameter.width_expression)
                if not 1 <= width <= 256:
                    reject(parameter.name, "width", "local width must be in 1..256")
            self.parameters[name] = parameter
            self.widths[name] = width
        elif statement.control_body is not None:
            self.scalar_control(statement, index)
        elif statement.within is not None:
            self.within(statement, index)
        elif token.text == "=":
            target = self.atom(args[0])
            if (
                self.helper_stack
                and not args[1].children
                and args[1].token.text in self.parameters
                and self.parameters[args[1].token.text].kind == "qnum"
                and args[1].token.text not in self.signed
                and self.parameters.get(target.text) is not None
                and self.parameters[target.text].role == "output"
                and self.parameters[target.text].kind == "qnum"
                and target.text not in self.signed
            ):
                view = self.array_view(args[1])
                register = self.register(target, len(view))
                if any(value == register for value, _ in view):
                    reject(target, "alias", "copy requires disjoint input and Output")
                self.effect_op("qreg.allocate", token, (register,))
                for bit, control in enumerate(view):
                    self.emit_indexed_x(token, (control, (register, bit)))
                return
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
        elif token.text == "$kickback_phase":
            condition, kickback_coefficient = args
            self.real_phase(
                token,
                Expression(replace(token, text="1")),
                kickback_coefficient,
                condition=condition,
                phase_sign=True,
            )
        elif token.text == "phase":
            default_coefficient = len(args) == 1
            if default_coefficient:
                args = (*args, Expression(Token("1.0", token.location)))
            if len(args) != 2 or "phase" in self.parameters:
                reject(
                    token,
                    "signature",
                    "phase requires an expression and unshadowed pi coefficient",
                )
            pending = [args[0]]
            indexed = False
            while pending:
                node = pending.pop()
                if node.token.text == "[":
                    indexed = True
                    break
                pending.extend(node.children)
            if (
                default_coefficient
                or bool(args[1].children)
                or args[1].token.text.isdecimal()
                or indexed
                or args[1].token.text == "["
                or any(
                    parameter.record is not None
                    for parameter in self.parameters.values()
                )
                or args[1].token.text in self.classical_values
                or args[1].token.text in self.classical_integers
                or (
                    self.helper_stack
                    and (
                        self.classical_values
                        or self.classical_arrays
                        or self.classical_integers
                        or self.integer_arrays
                    )
                )
            ):
                self.real_phase(token, args[0], args[1])
                return
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
            if not (
                lower_indexed_parity(self, token, args[0], args[1])
                or lower_fixed_xor(self, token, args[0], args[1])
            ):
                self.xor_update(token, args[0], args[1])
        elif token.text in {
            "encode_in_angle",
            "encode_on_bloch",
            "inplace_encode_on_bloch",
        }:
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
                if (
                    args[0].token.text == "."
                    and len(args[0].children) == 2
                    and args[0].children[1].token.text in {"size", "len"}
                ):
                    size = self.static_integer(args[0])
                elif not args[0].children:
                    name = args[0].token.text
                    size = self.classical_integers.get(name)
                    if size is None:
                        size = exact_unsigned_literal(name)
                elif args[0].token.text in {"+", "-", "*", "%", "**", "$ceil_log2"}:
                    size = self.static_integer(args[0])
                if size is None:
                    reject(
                        args[0].token,
                        "width",
                        "allocation size must be a literal or record size",
                    )
                if not 1 <= size <= 256:
                    reject(
                        args[0].token,
                        "width",
                        "allocation size must be a literal in 1..256",
                    )
            target = self.atom(args[-1])
            if self.helper_stack and (
                self.parameters.get(target.text) is None
                or not (
                    self.parameters[target.text].role == "output"
                    or self.parameters[target.text].role == "local"
                    and self.parameters[target.text].kind == "qbit"
                )
            ):
                reject(target, "free", "bound allocation requires a local qbit")
            phase_allocation = infer_phase_allocation(self, token, args, size)
            phase_parameter = self.parameters.get(target.text)
            destination = self.register(
                target,
                size,
                allow_qubit_array=True,
                allocate_local=True,
                allow_fractional_output=(
                    phase_parameter is not None
                    and phase_parameter.kind in {"qphase", "qphase_signed"}
                    and target.text not in self.fractional_bits
                ),
            )
            if size is not None and self.widths[target.text] != size:
                reject(
                    target,
                    "width",
                    "allocation size must agree with the declared width",
                )
            if len(args) == 4 and not phase_allocation:
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
                elif target.text in self.signed:
                    reject(
                        sign.token,
                        "type",
                        "allocation signedness must agree with the declared type",
                    )
            self.effect_op("qreg.allocate", token, (destination,))
        elif token.text in {
            "prepare_amplitudes",
            "inplace_prepare_amplitudes",
            "prepare_state",
        }:
            self.prepare_state_vector(token, args)
        elif token.text == "prepare_exponential_state":
            self.prepare_exponential_state(token, args)
        elif token.text == "modular_exp":
            if len(args) != 4 or any(
                arg.children or not arg.token.text.isdecimal() for arg in args[:2]
            ):
                reject(
                    token,
                    "signature",
                    "modular_exp requires literal modulus and base, target and exponent",
                )
            modulus, base = (int(arg.token.text) for arg in args[:2])
            names = tuple(self.atom(arg) for arg in args[2:])
            registers = tuple(self.register(name) for name in names)
            if any(
                self.parameters[name.text].kind != "qnum" or name.text in self.signed
                for name in names
            ):
                reject(token, "type", "modular_exp requires unsigned qnum registers")
            self.effect_op(
                "qreg.modular_exp", token, registers, modulus=modulus, base=base
            )
        elif token.text == "free":
            if len(args) != 1:
                reject(token, "signature", "free requires one local qbit")
            lower_free(self, self.atom(args[0]))
        elif token.text == "drop":
            if len(args) != 1 or self.scope != "main" or self.local_scope is not None:
                reject(token, "drop", "drop requires one local register in main")
            target = self.atom(args[0])
            drop_parameter = self.parameters.get(target.text)
            if (
                drop_parameter is None
                or drop_parameter.role != "local"
                or drop_parameter.kind not in {"qbit", "qnum", "qbit[]", "qphase"}
            ):
                reject(target, "drop", "drop requires a local register")
            destination = self.register(
                target,
                allow_qubit_array=True,
                allow_fractional_output=drop_parameter.kind == "qphase",
            )
            self.effect_op("qreg.drop", token, (destination,))
            self.dropped.add(target.text)
        elif token.text == "assign_phase_table":
            if len(args) != 2:
                reject(
                    token, "signature", "phase table requires angles and a quantum word"
                )
            angles = self.real_array(args[0])
            locations = self.array_view(args[1])
            if not 1 <= len(locations) <= 7 or len(angles) != 1 << len(locations):
                reject(
                    token,
                    "width",
                    "phase table size must match a one-to-seven-bit word",
                )
            if len(locations) != len(set(locations)):
                reject(token, "alias", "phase table target locations must be disjoint")
            registers, indices = zip(*locations, strict=True)
            for word, angle in enumerate(angles):
                if abs(angle) > 2**53:
                    reject(
                        token, "phase", "phase table angles must fit finite phase range"
                    )
                if angle or word == 0:
                    self.effect_op(
                        "qreg.control_phase",
                        token,
                        registers,
                        indices=indices,
                        control_state=word,
                        angle=angle,
                    )
        elif token.text == "IDENTITY":
            if len(args) != 1:
                reject(token, "signature", "IDENTITY requires one register")
            indexed = args[0].token.text == "[" and not args[0].children[0].children
            locations = (
                (self.gate_target(args[0]),) if indexed else self.array_view(args[0])
            )
            root = args[0].children[0] if indexed else args[0]
            port = self.parameters.get(root.token.text) if not root.children else None
            checked: set[Value] = set()
            for value, bit in locations:
                if value not in checked:
                    if port is not None and port.role == "const":
                        self.effect_op(
                            "qreg.control_phase",
                            token,
                            (value,),
                            indices=(bit,),
                            angle=0.0,
                        )
                    else:
                        self.effect_op(
                            "qreg.gate", token, (value,), gate="i", index=bit
                        )
                    checked.add(value)
        elif token.text == "qft":
            if len(args) != 1:
                reject(token, "signature", "qft requires one qubit array")
            target = self.atom(args[0])
            fourier_parameter = self.parameters.get(target.text)
            kind = fourier_parameter.kind if fourier_parameter is not None else None
            if target.text in self.fractional_bits:
                reject(target, "precision", "a fractional result is a terminal output")
            register = self.register(
                target, allow_qubit_array=True, allow_fractional_output=kind == "qphase"
            )
            if kind not in {"qbit[]", "qnum", "qphase"}:
                reject(target, "type", "qft requires a qubit array or integer register")
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
            destination = self.register(self.atom(args[-1]), allow_qubit_array=True)
            self.effect_op(
                "qreg.uniform_interval", token, (destination,), start=start, stop=stop
            )
        elif token.text == "apply_to_all":
            if (
                len(args) != 2
                or args[0].children
                or args[0].token.text not in {"H", "X", "Z"}
            ):
                reject(
                    token,
                    "signature",
                    "apply_to_all currently requires H, X or Z and a register",
                )
            for value, wire in self.array_view(args[1]):
                self.effect_op(
                    "qreg.gate",
                    token,
                    (value,),
                    gate=args[0].token.text.lower(),
                    index=wire,
                )
        elif token.text == "unitary":
            lower_unitary(self, token, args)
        elif token.text == "linear_pauli_rotations":
            self.linear_pauli_rotations(token, args)
        elif token.text == "single_pauli_exponent":
            self.single_pauli_exponent(token, args)
        elif token.text in {"suzuki_trotter", "multi_suzuki_trotter"}:
            self.suzuki_trotter(token, args)
        elif token.text == "exponentiate":
            self.exponentiate(token, args)
        elif token.text == "qdrift":
            self.qdrift(token, args)
        elif token.text == "RZZ":
            if len(args) != 2:
                reject(token, "signature", "RZZ requires an angle and a qubit array")
            angle = self.angle(args[0], token)
            view = self.array_view(args[1])
            if len(view) != 2:
                reject(args[1].token, "type", "RZZ requires a two-qubit array or view")
            if len(set(view)) != 2:
                reject(args[1].token, "alias", "RZZ requires distinct qubit locations")
            (first, first_index), (second, second_index) = view
            self.effect_op(
                "qreg.pair_gate",
                token,
                (first, second),
                gate="rzz",
                indices=(first_index, second_index),
                angle=angle,
            )
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
            locations = tuple(
                self.gate_target(arg, allow_numeric=token.text in {"CX", "CZ"})
                for arg in targets
            )
            registers, indices = zip(*locations, strict=True)
            if pair and (
                any(arg.token.text == "[" for arg in targets)
                or any(register_width(register) != 1 for register in registers)
            ):
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
                if args[0].token.text in {".", "["}:
                    gate_locations = self.array_view(args[0])
                else:
                    target = self.atom(args[0])
                    if target.text in self.views:
                        gate_locations = self.array_view(args[0])
                    else:
                        if target.text in self.fractional_bits:
                            reject(
                                target,
                                "precision",
                                "a fractional result is a terminal output",
                            )
                        hadamard_parameter = self.parameters.get(target.text)
                        destination = self.register(
                            target,
                            allow_qubit_array=True,
                            allow_fractional_output=hadamard_parameter is not None
                            and hadamard_parameter.kind == "qphase",
                        )
                        gate_locations = tuple(
                            (destination, bit)
                            for bit in range(register_width(destination))
                        )
            else:
                gate_locations = (
                    self.gate_target(
                        args[0], allow_numeric=token.text in {"H", "X", "Z"}
                    ),
                )
            for destination, bit in gate_locations:
                if token.text == "Y":
                    for gate, gate_attributes in (
                        ("s", {"inverse": True}),
                        ("x", {}),
                        ("s", {}),
                    ):
                        self.effect_op(
                            "qreg.gate",
                            token,
                            (destination,),
                            gate=gate,
                            index=bit,
                            **gate_attributes,
                        )
                    continue
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
        statements = self.syntax.statements
        for index, statement in enumerate(statements):
            if self.scope == "main" and index + 4 == len(statements):
                if lower_cnf_kickback(self, statements[index:]):
                    break
            self.statement(statement, str(index))
        if any(
            parameter.role == "output" and name in self.suspended
            for name, parameter in self.parameters.items()
        ):
            reject(self.syntax.name, "binding", "restore every output before return")
        if any(
            parameter.role == "local"
            and name in self.views
            and name not in self.suspended
            for name, parameter in self.parameters.items()
        ):
            reject(
                self.syntax.name,
                "binding",
                "merge or restore every local view before return",
            )
        output_parameters = tuple(
            p for p in self.syntax.parameters if p.role == "output"
        )
        borrowed = tuple(
            self.register(p.name) for p in self.syntax.parameters if p.role == "const"
        )
        for parameter in output_parameters:
            kind = self.parameters[parameter.name.text].kind
            if (
                kind == "qphase_signed"
                and parameter.name.text not in self.fractional_bits
            ):
                reject(
                    parameter.name,
                    "qpe",
                    "signed fixed-point output requires phase estimation or a checked writable bit view",
                )
            if kind == "qphase":
                width = self.widths[parameter.name.text]
                assert width is not None
                self.fractional_bits.setdefault(parameter.name.text, width)
        declared = tuple(
            self.register(p.name, allow_qubit_array=True, allow_fractional_output=True)
            for p in output_parameters
        )
        declarations = tuple(
            Operation(
                "qreg.declare",
                results=(value,),
                attributes={
                    "name": parameter.name.text,
                    **(
                        {
                            "record_type": parameter.record,
                            "fields": tuple(
                                (
                                    (
                                        field.name.text,
                                        field.kind,
                                        field.width,
                                        field.length,
                                    )
                                    if field.length is not None
                                    else (
                                        field.name.text,
                                        (
                                            "qnum"
                                            if field.kind == "qnum_signed"
                                            else field.kind
                                        ),
                                        field.width,
                                    )
                                )
                                for field in self.record_types[parameter.record].leaves(
                                    self.record_types
                                )
                            ),
                        }
                        if parameter.record is not None
                        else {}
                    ),
                },
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


def _qaoa_workflow(
    settings: VqeSettings, main: Function, constants: Mapping[str, Expression]
) -> QaoaWorkflow:
    token = Token("cscope", settings.location)
    inputs = [p for p in main.parameters if p.role == "real" and p.kind == "real[]"]
    outputs = [p for p in main.parameters if p.role == "output" and p.kind == "qbit[]"]
    if (
        len(main.parameters) != 2
        or len(inputs) != 1
        or len(outputs) != 1
        or inputs[0].width != len(settings.initial_point)
        or outputs[0].width is None
        or len(main.statements) != 2
    ):
        reject(
            token,
            "cscope",
            "VQE requires a sized QAOA parameter vector and bit-array output",
        )
    initial, ansatz = main.statements
    parameter, output = inputs[0].name.text, outputs[0].name.text
    width = outputs[0].width
    assert width is not None
    if (
        initial.token.text != "allocate"
        or not initial.arguments
        or initial.arguments[-1].children
        or initial.arguments[-1].token.text != output
        or ansatz.token.text != "qaoa_penalty"
        or len(ansatz.arguments) != 4
        or any(arg.children for arg in ansatz.arguments[1:])
        or tuple(arg.token.text for arg in ansatz.arguments[1:])
        != (parameter, settings.operator, output)
    ):
        reject(
            token,
            "cscope",
            "VQE requires fresh QAOA preparation with its declared Hamiltonian",
        )
    operator = constants.get(settings.operator)
    if operator is None or operator.token.text != "PauliTerm[]":
        reject(token, "cscope", "VQE requires a declared PauliTerm[] Hamiltonian")
    diagonal = []
    for term in operator.children:
        axes, coefficient = term.children
        support = _pauli_support(axes, width, sparse=False)
        if any(axis != "Z" for _, axis in support):
            reject(
                axes.token, "cscope", "exact CVaR requires a diagonal I/Z Hamiltonian"
            )
        negative = coefficient.token.text == "-"
        magnitude = coefficient.children[0] if negative else coefficient
        value = _literal_angle(magnitude, token)
        diagonal.append(
            (sum(1 << bit for bit, _ in support), -value if negative else value)
        )
    return QaoaWorkflow(settings, parameter, output, width, tuple(diagonal))


def parse_qmod(
    source: str,
    *,
    filename: str = "<qmod>",
    source_version: str,
    classical_inputs: Mapping[str, float | Sequence[float]] | None = None,
    machine_precision: int | None = None,
) -> QmodImport:
    """Adapt the pinned unsigned scalar native subset to verified owned records.

    The caller declares source_version; this does not authenticate source bytes.
    Originals are independently checked against their retained source manifest.
    No files, SDKs, synthesis, execution, eval or generated Python are involved.
    """
    source, settings = split_cscope(source, filename)
    tokens = tokenize(source, filename)
    check_source_version(source_version, tokens[0])
    definitions: dict[str, Function] = {}
    parser = Parser(tokens)
    token_names = {token.text for token in tokens}
    parser.closed_fixed_point = (
        machine_precision is not None
        and "amplitude_amplification" in token_names
        and bool({"SIGNED", "True"} & token_names)
    )
    functions_syntax, declarations = parser.parse()
    constants: dict[str, Expression] = {}
    constant_integers: dict[str, int] = {}
    for name_token, operator in declarations:
        name = name_token.text
        if name in constants or name in _INTRINSICS or name == "main":
            reject(name_token, "name", "duplicate or reserved top-level constant")
        if name in parser.integer_constants:
            constant_integers[name] = parser.integer_values[name]
            constants[name] = operator
            continue
        if name in parser.matrix_constants:
            validate_matrix_constant(name_token, operator)
            constants[name] = operator
            continue
        if operator.token.text == "PauliTerm[]":
            if not operator.children:
                reject(operator.token, "pauli", "PauliTerm[] constant cannot be empty")
            width = len(operator.children[0].children[0].children)
            if not 1 <= width <= 256:
                reject(operator.token, "pauli", "Pauli word width must be in 1..256")
            for term in operator.children:
                axes, coefficient = term.children
                _pauli_support(axes, width, sparse=False)
                if coefficient.token.text == "-":
                    coefficient = coefficient.children[0]
                _literal_angle(coefficient, name_token)
            constants[name] = operator
            continue
        if operator.token.text != "SparsePauliOp":
            if operator.token.text == "[":
                for item in operator.children:
                    _preparation_value(item)
            else:
                _preparation_value(operator)
            constants[name] = operator
            continue
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
            reject(syntax.name, "name", "function shadows a top-level constant")
        for parameter in syntax.parameters:
            if parameter.name.text in constants and parameter.role != "operator":
                reject(parameter.name, "name", "parameter shadows a top-level constant")
        definitions[name] = syntax
    if "main" not in definitions:
        reject(tokens[0], "entry", "source requires one qfunc main")
    fixed_program: FixedPreparation | None = None
    if parser.closed_fixed_point:
        assert machine_precision is not None
        if not 1 <= machine_precision <= 8:
            reject(tokens[0], "precision", "machine precision must be in 1..8")
        root = definitions["main"]
        declared = {
            p.name.text: p for p in root.parameters if p.role in {"real", "integer"}
        }
        supplied = classical_inputs or {}
        if (
            not isinstance(supplied, Mapping)
            or set(declared) != set(supplied)
            or any(p.kind != "int" for p in declared.values())
        ):
            reject(
                root.name, "binding", f"classical inputs must match {sorted(declared)}"
            )
        integer_values = {
            name: checked_integer(supplied[name], port.name)
            for name, port in declared.items()
        }
        fixed_program = build_fixed_preparation(
            definitions, parser.records, machine_precision, integer_values
        )
        definitions = {"main": replace(root, statements=())}
    numeric_program = try_numeric_program(definitions)
    if numeric_program is not None:
        definitions = {"main": replace(definitions["main"], statements=())}
    closed_region = None
    if {"skip_control", "qft_const_adder_phase"} <= token_names:
        closed_region = try_closed_region(definitions, parser.records)
        if closed_region is not None:
            definitions = {"main": replace(definitions["main"], statements=())}
    definitions, deferred_measurements = defer_private_measurements(definitions)
    definitions = fold_skip_control_tables(definitions)
    definitions = expand_amplification(definitions, constants)
    definitions = expand_iterations(definitions)
    definitions = specialize_quantum_callbacks(definitions, constants)
    definitions = expand_nested_array_kickbacks(definitions)
    definitions = fuse_consuming_preparations(definitions)
    definitions = prove_annotated_selections(definitions)

    def compile_probe(
        function: Function, proof_definitions: Mapping[str, Function]
    ) -> HybridProgram:
        def resolve_probe(token: Token) -> tuple[Function, HybridProgram]:
            reject(
                token, "ownership", "selector proof requires statically bound helpers"
            )

        probe = _Adapt(
            probe_source(function),
            resolve_probe,
            proof_definitions,
            constants,
            parser.records,
        )
        probe.classical_integers.update(constant_integers)
        return probe.adapt()

    definitions = normalize_clean_selectors(definitions, compile_probe)
    definitions = expand_borrowed_phase_views(definitions, constants)
    definitions = expand_phase_kickbacks(definitions)
    verify_permutations(definitions)
    fixed_helpers = payoff_helpers(definitions)
    definitions = expand_predicate_oracles(
        definitions, constants, parser.records, inferred_helpers=fixed_helpers
    )
    definitions = expand_static_oracles(definitions, constants, parser.records)
    main = definitions["main"]

    def fractional_record(record: str) -> bool:
        return any(
            field.fractional_bits
            or (field.record is not None and fractional_record(field.record))
            for field in parser.records[record].fields
        )

    if any(
        port.record is not None and fractional_record(port.record)
        for port in main.parameters
    ):
        reject(
            main.name,
            "precision",
            "fractional record fields require checked fixed-point lowering",
        )
    if any(port.fractional_bits for port in main.parameters):
        reject(
            main.name,
            "precision",
            "fixed-point outputs require a checked synthesis path",
        )
    workflow = (
        _qaoa_workflow(settings, main, constants) if settings is not None else None
    )
    parameters = {
        p.name.text: p for p in main.parameters if p.role in {"real", "integer"}
    }
    if classical_inputs is None and workflow is not None:
        classical_inputs = {workflow.parameter: workflow.settings.initial_point}
    inputs = {} if classical_inputs is None else classical_inputs
    if not isinstance(inputs, Mapping) or set(inputs) != set(parameters):
        reject(
            main.name, "binding", f"classical inputs must match {sorted(parameters)}"
        )
    bindings: dict[str, float] = {}
    arrays: dict[str, tuple[float, ...]] = {}
    integer_bindings: dict[str, int] = {}
    integer_arrays: dict[str, tuple[int, ...]] = {}

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
        if parameters[name].role == "integer":
            if parameters[name].kind == "int":
                integer_bindings[name] = checked_integer(value, parameters[name].name)
            else:
                integer_arrays[name] = checked_integer_array(
                    value, parameters[name].name, parameters[name].width
                )
            continue
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
        if name == "main" or not _bound_helper(syntax) or name in fixed_helpers:
            continue
        if any(port.kind == "SparsePauliOp" for port in syntax.parameters):
            power_action(syntax)
            continue
        if slice_output(syntax) is not None:
            continue
        if len({port.name.text for port in syntax.parameters}) != len(
            syntax.parameters
        ):
            reject(syntax.name, "name", "duplicate helper parameter")
        if any(port.fractional_bits for port in syntax.parameters):
            if (
                computed_output(syntax) is None
                and reinterpreted_selector(syntax, definitions) is None
            ):
                reject(
                    syntax.name,
                    "precision",
                    "fixed-point helpers require a checked computed rotation",
                )
            continue
        fixed_locals = [
            position
            for position, statement in enumerate(syntax.statements)
            if statement.declaration is not None
            and statement.declaration.fractional_bits
        ]
        if fixed_locals and (
            fixed_locals != [len(syntax.statements) - 3]
            or composed_rotation(syntax.statements[-3:], definitions) is None
        ):
            reject(
                syntax.name,
                "precision",
                "fixed-point locals require a checked computed rotation",
            )
        if any(
            port.role not in {"real", "integer", "inout", "callable"}
            and port.kind not in {"PauliTerm[]", "IndexedPauli[]"}
            and not (
                port.role == "output"
                and port.kind in {"qbit", "qbit[]", "qphase", "qnum"}
            )
            and not (
                port.role == "const"
                and (
                    port.kind == "qphase"
                    or (
                        port.kind in {"qbit", "qnum", "qnum_signed"}
                        and port.record is None
                    )
                    or (
                        port.record is not None
                        and len(syntax.statements) == 1
                        and syntax.statements[0].token.text == "$projected_rz"
                    )
                    or (
                        port.kind == "qbit[]"
                        and port.record is None
                        and not syntax.permutation
                        and any(p.role == "inout" for p in syntax.parameters)
                    )
                    or is_const_cx_permutation(syntax)
                    or is_selector_helper(syntax)
                    or (
                        port.kind == "qbit[]"
                        and port.record is None
                        and any(p.kind == "IndexedPauli[]" for p in syntax.parameters)
                    )
                )
            )
            for port in syntax.parameters
        ):
            reject(
                syntax.name,
                "role",
                "bound helpers require real or writable qubit array ports",
            )
        local_bits = {
            statement.declaration.name.text
            for statement in syntax.statements
            if statement.declaration is not None
            and statement.declaration.kind == "qbit"
        }
        output_names = {
            port.name.text for port in syntax.parameters if port.role == "output"
        }
        snapshot = snapshot_start(syntax.statements)
        for ordinal, statement in enumerate(syntax.statements):
            if snapshot is not None and ordinal in {snapshot, snapshot + 1}:
                continue
            if statement.token.text in {"allocate", "free"} and (
                not statement.arguments
                or statement.arguments[-1].children
                or statement.arguments[-1].token.text
                not in (
                    local_bits | output_names
                    if statement.token.text == "allocate"
                    else local_bits
                )
            ):
                reject(
                    statement.token,
                    "scope",
                    "bound allocation and free require a local qbit",
                )
            declaration = statement.declaration
            if declaration is not None and declaration.kind in {
                "qbit",
                "qbit[]",
                "qnum",
            }:
                continue
            controlled_qft = (
                statement.control_body is not None
                and len(statement.control_body) == 1
                and statement.control_body[0].token.text == "qft"
            )
            hadamard_within = (
                statement.within is not None
                and len(statement.within) == 1
                and statement.within[0].token.text == "H"
            )
            called = definitions.get(statement.token.text)
            bound_call = (
                called is not None
                and called.name.text != "main"
                and _bound_helper(called)
            ) or any(
                port.role == "callable" and port.name.text == statement.token.text
                for port in syntax.parameters
            )
            copy_output = (
                statement.token.text == "="
                and not any(arg.children for arg in statement.arguments)
                and statement.arguments[0].token.text in output_names
                and any(
                    port.name.text == statement.arguments[1].token.text
                    and port.kind == "qnum"
                    and port.role in {"const", "inout"}
                    for port in syntax.parameters
                )
            )
            if statement.declaration is not None or (
                statement.token.text
                not in _STATIC_GATES
                | EVOLUTION_CALLS
                | {
                    "swap_test",
                    "grover_diffuser",
                    "multiswap",
                    "unitary",
                    "+=",
                    "^=",
                    "phase",
                    "$projected_rz",
                    "$kickback_phase",
                    "inplace_prepare_state",
                    "inplace_prepare_amplitudes",
                    "prepare_amplitudes",
                    "select_rotation",
                    "linear_pauli_rotations",
                    "allocate",
                    "free",
                    "->",
                }
                and not copy_output
                and not controlled_qft
                and not is_modular_power(statement)
                and not hadamard_within
                and not is_binding_scope(statement)
                and not is_reversible_scope(statement)
                and statement.repeat_index is None
                and statement.static_if_body is None
                and statement.map_callback is None
                and statement.control_body is None
                and statement.inverse_body is None
                and statement.callback_index is None
                and not bound_call
            ):
                reject(
                    statement.token,
                    "scope",
                    "bound helper bodies admit only gates, bounded unitary and Pauli evolution, and scoped QFT",
                )

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
            adapter = _Adapt(syntax, resolve, definitions, constants, parser.records)
            adapter.classical_integers.update(constant_integers)
            if name == "main":
                adapter.classical_values.update(bindings)
                adapter.classical_arrays.update(arrays)
                adapter.classical_integers.update(integer_bindings)
                adapter.integer_arrays.update(integer_arrays)
                if numeric_program is not None:
                    emit_numeric_program(adapter, numeric_program)
                if closed_region is not None:
                    emit_closed_region(adapter, closed_region)
                if fixed_program is not None:
                    emit_fixed_preparation(adapter, fixed_program)
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
        tuple(
            name
            for p in main.parameters
            for name in (
                (p.name.text,)
                if p.name.text in adapter.signed
                else (
                    tuple(
                        f"{p.name.text}.{field.name.text}"
                        for field in parser.records[p.record].leaves(parser.records)
                        if field.kind == "qnum_signed"
                    )
                    if p.record is not None
                    else ()
                )
            )
        ),
        workflow,
        deferred_measurements,
    )
