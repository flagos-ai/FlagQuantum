"""Bounded algebraic normal forms for exact unsigned value candidates.

Each polynomial is a set of input-bit masks. Symmetric difference adds terms
over GF(2), and mask union multiplies square-free monomials (b*b = b).
Ripple addition and partial products expand only within the fixed bit budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ...core.ir import CircuitIR, Instruction
from .._hybrid.model import HybridProgram, ValueId
from .arithmetic_rewrites import normalize_arithmetic
from .bindings import verify_bindings
from .lifetimes import analyze_lifetimes
from .profile import _positive_integer

_Poly = frozenset[int]
_Bits = tuple[_Poly, ...]
_ZERO: _Poly = frozenset()
_ONE: _Poly = frozenset({0})
_MAX_INPUT_BITS = 6
_MAX_WIDTH = 12
_MAX_OPERATIONS = 64
_MAX_PRODUCTS = 20_000


class BooleanCandidateError(ValueError):
    """The optional Boolean family cannot supply a candidate."""


@dataclass(frozen=True)
class BooleanValueCandidate:
    circuit: CircuitIR
    output_terms: _Bits
    input_wires: tuple[int, ...]
    workspace: tuple[int, ...]
    source_result: ValueId


class _Expansion:
    def __init__(self) -> None:
        self.products_left = _MAX_PRODUCTS

    def multiply(self, left: _Poly, right: _Poly) -> _Poly:
        self.products_left -= len(left) * len(right)
        if self.products_left < 0:
            raise BooleanCandidateError("budget: Boolean term products")
        parity: set[int] = set()
        for a in left:
            for b in right:
                parity.symmetric_difference_update((a | b,))
        return frozenset(parity)

    def add_words(self, left: _Bits, right: _Bits, width: int) -> _Bits:
        carry = _ZERO
        bits = []
        for index in range(width):
            a = left[index] if index < len(left) else _ZERO
            b = right[index] if index < len(right) else _ZERO
            pair = a ^ b
            bits.append(pair ^ carry)
            if index + 1 < width:
                carry = self.multiply(a, b) ^ self.multiply(pair, carry)
        return tuple(bits)

    def multiply_words(self, left: _Bits, right: _Bits, width: int) -> _Bits:
        result: _Bits = ()
        for shift, bit in enumerate(right[:width]):
            if bit:
                partial = (_ZERO,) * shift + tuple(
                    self.multiply(term, bit) for term in left[: width - shift]
                )
                result = self.add_words(result, partial, width)
        return result or (_ZERO,) * width


def lower_boolean_value(
    functions: Sequence[HybridProgram],
    inputs: Mapping[ValueId, Sequence[int]],
    output: Sequence[int],
    *,
    entry: str,
    n_qubits: int,
    workspace: Sequence[int] = (),
    workspace_state: Literal[
        "fresh_zero", "caller_asserted_zero", "unknown"
    ] = "unknown",
    max_operations: int = 256,
) -> BooleanValueCandidate:
    """Produce an optional XOR candidate for AO-04, without selecting it."""
    _positive_integer(max_operations, "max_operations")
    verify_bindings(
        functions,
        inputs,
        output,
        entry=entry,
        n_wires=n_qubits,
        workspace=workspace,
        workspace_state=workspace_state,
    )
    if sum(map(len, inputs.values())) > _MAX_INPUT_BITS or len(output) > _MAX_WIDTH:
        raise BooleanCandidateError("budget: Boolean input or output width")

    analysis = analyze_lifetimes(functions, entry=entry)
    original = analysis.program.body.blocks[0]
    if len(original.operations) > _MAX_OPERATIONS:
        raise BooleanCandidateError("budget: Boolean source operations")
    supported = {"arith.constant", "qint.add", "qint.multiply"}
    if any(op.name not in supported for op in original.operations[:-1]):
        raise BooleanCandidateError("unsupported: expected a pure unsigned expression")
    normalized = normalize_arithmetic(analysis)
    block = normalized.program.body.blocks[0]
    wires: list[int] = []
    values: dict[ValueId, _Bits] = {}
    for arg in block.arguments:
        argument_bits = []
        for wire in reversed(inputs[arg.id]):
            argument_bits.append(frozenset({1 << len(wires)}))
            wires.append(wire)
        values[arg.id] = tuple(argument_bits)

    expand = _Expansion()
    for op in block.operations[:-1]:
        key = op.results[0].id
        evidence = normalized.facts[key]
        width = evidence.effective_width
        if width > _MAX_WIDTH:
            raise BooleanCandidateError("budget: Boolean intermediate width")
        constant = evidence.constant
        if constant is not None:
            values[key] = tuple(
                _ONE if constant & (1 << bit) else _ZERO for bit in range(width)
            )
        else:
            a, b = (values[value.id] for value in op.operands)
            if op.name == "qint.add":
                values[key] = expand.add_words(a, b, width)
            else:
                values[key] = expand.multiply_words(a, b, width)

    result = block.operations[-1].operands[0].id
    bits = values[result]
    bits += (_ZERO,) * (len(output) - len(bits))
    used_workspace: tuple[int, ...] = ()
    gates: list[Instruction] = []
    for target, terms in zip(reversed(output), bits, strict=True):
        for mask in sorted(terms):
            controls = tuple(w for i, w in enumerate(wires) if mask & (1 << i))
            degree = len(controls)
            if degree > 3 or (degree == 3 and not workspace):
                raise BooleanCandidateError(
                    "mapping: requires more workspace or factoring"
                )
            if len(gates) + (3 if degree == 3 else 1) > min(max_operations, 256):
                raise BooleanCandidateError("budget: Boolean mapped operations")
            if degree == 3:
                scratch = workspace[0]
                used_workspace = (scratch,)
                compute = Instruction("ccx", (*controls[:2], scratch))
                gates.extend(
                    (
                        compute,
                        Instruction("ccx", (scratch, controls[2], target)),
                        compute,
                    )
                )
            else:
                gates.append(
                    Instruction(("x", "cx", "ccx")[degree], (*controls, target))
                )
    return BooleanValueCandidate(
        CircuitIR(n_qubits, tuple(gates)),
        bits,
        tuple(wires),
        used_workspace,
        original.operations[-1].operands[0].id,
    )
