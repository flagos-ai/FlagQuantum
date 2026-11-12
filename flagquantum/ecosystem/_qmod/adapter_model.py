"""Numeric and primitive Qmod adapter contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from operator import eq, ge, gt, le, lt, ne
from typing import NoReturn

from ...compiler import _pauli_schedule
from ...compiler._construction.numeric_expression import WordExpression
from ...compiler._hybrid.model import (
    HybridProgram,
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


class _Expression(WordExpression[Expression]):
    """Bind source locations and syntax to Compiler-owned numeric semantics."""

    def __init__(
        self,
        node: Expression,
        widths: Mapping[str, int | None],
        name: str,
        signed_names: set[str],
    ) -> None:
        def fail(node: Expression, code: str, message: str) -> NoReturn:
            reject(node.token, code, message)

        super().__init__(
            node,
            widths,
            name,
            signed_names,
            text=lambda item: item.token.text,
            children=lambda item: item.children,
            location=lambda item: item.token.location,
            reject=fail,
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
