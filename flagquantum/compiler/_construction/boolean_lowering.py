"""Bounded algebraic normal forms for exact unsigned value candidates.

Each polynomial is a set of input-bit masks. Symmetric difference adds terms
over GF(2), and mask union multiplies square-free monomials (b*b = b).
Ripple addition and partial products expand only within the fixed bit budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ...core.controlled import _clean_mcx_instructions
from ...core.ir import CircuitIR, Instruction
from .._hybrid.model import HybridProgram, ValueId
from .arithmetic_rewrites import normalize_arithmetic
from .bindings import verify_bindings
from .lifetimes import LifetimeAnalysis, analyze_lifetimes
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

    def __init__(self, message: str, *, exhausted: bool = False) -> None:
        super().__init__(message)
        self.exhausted = exhausted


@dataclass(frozen=True)
class BooleanValueCandidate:
    circuit: CircuitIR
    output_terms: _Bits
    input_wires: tuple[int, ...]
    workspace: tuple[int, ...]
    source_result: ValueId
    lifetimes: LifetimeAnalysis


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

    def bitwise_words(
        self, operator: str, operands: tuple[_Bits, ...], width: int
    ) -> _Bits:
        words = tuple(
            word[:width] + (_ZERO,) * max(0, width - len(word)) for word in operands
        )
        if operator == "~":
            return tuple(bit ^ _ONE for bit in words[0])
        left, right = words
        if operator == "^":
            return tuple(a ^ b for a, b in zip(left, right, strict=True))
        products = tuple(self.multiply(a, b) for a, b in zip(left, right, strict=True))
        if operator == "&":
            return products
        return tuple(
            a ^ b ^ product for a, b, product in zip(left, right, products, strict=True)
        )

    def equal_words(self, left: _Bits, right: _Bits) -> _Bits:
        width = max(len(left), len(right))
        equal = _ONE
        for mismatch in self.bitwise_words("^", (left, right), width):
            equal = self.multiply(equal, mismatch ^ _ONE)
        return (equal,)

    def less_words(self, left: _Bits, right: _Bits) -> _Bits:
        # Visit from least to most significant: the highest mismatch wins.
        width = max(len(left), len(right))
        left += (_ZERO,) * (width - len(left))
        right += (_ZERO,) * (width - len(right))
        less = _ZERO
        for a, b in zip(left, right, strict=True):
            less ^= self.multiply(a ^ b, less ^ b)
        return (less,)

    def select_words(
        self, predicate: _Poly, yes: _Bits, no: _Bits, width: int
    ) -> _Bits:
        yes += (_ZERO,) * (width - len(yes))
        no += (_ZERO,) * (width - len(no))
        return tuple(
            b ^ self.multiply(predicate, a ^ b)
            for a, b in zip(yes[:width], no[:width], strict=True)
        )


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
    if len(output) > _MAX_WIDTH:
        raise BooleanCandidateError("budget: Boolean input or output width")

    analysis = analyze_lifetimes(functions, entry=entry)
    original = analysis.program.body.blocks[0]
    if len(original.operations) > _MAX_OPERATIONS:
        raise BooleanCandidateError("budget: Boolean source operations")
    supported = {"arith.constant", "qint.add", "qint.multiply"}
    if any(op.name == "qint.bitwise" for op in original.operations[:-1]):
        supported.update({"qint.bitwise", "qint.equal"})
    if any(op.name == "qint.select" for op in original.operations[:-1]):
        supported.update({"qint.select", "qint.less", "qint.equal"})
    if any(
        op.name not in supported or any(op.attributes.get("signed", ()))
        for op in original.operations[:-1]
    ):
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
            operands = tuple(values[value.id] for value in op.operands)
            if op.name == "qint.bitwise":
                values[key] = expand.bitwise_words(
                    op.attributes["operator"], operands, width
                )
            elif op.name == "qint.select":
                predicate, yes, no = operands
                values[key] = expand.select_words(predicate[0], yes, no, width)
            else:
                a, b = operands
                if op.name == "qint.add":
                    values[key] = expand.add_words(a, b, width)
                elif op.name == "qint.equal":
                    values[key] = expand.equal_words(a, b)
                elif op.name == "qint.less":
                    values[key] = expand.less_words(a, b)
                else:
                    values[key] = expand.multiply_words(a, b, width)

    result = block.operations[-1].operands[0].id
    bits = values[result]
    bits += (_ZERO,) * (len(output) - len(bits))
    support = 0
    for terms in bits:
        for mask in terms:
            support |= mask
    if support.bit_count() > _MAX_INPUT_BITS:
        raise BooleanCandidateError(
            "budget: Boolean input or output width (active inputs)"
        )
    used_workspace: tuple[int, ...] = ()
    gates: list[Instruction] = []
    for target, terms in zip(reversed(output), bits, strict=True):
        active = sorted(
            {i for mask in terms for i in range(len(wires)) if mask >> i & 1}
        )
        # A signed Boolean cube has exactly one satisfying assignment on its
        # support. Recognize it from the bounded ANF, independently of syntax.
        if any(mask.bit_count() > 3 for mask in terms):
            satisfying = []
            for assignment in range(1 << len(active)):
                basis = sum(
                    1 << bit for i, bit in enumerate(active) if assignment >> i & 1
                )
                if sum(mask & basis == mask for mask in terms) % 2:
                    satisfying.append(assignment)
            if len(satisfying) == 1:
                controls = tuple(wires[bit] for bit in active)
                cube_scratch = tuple(workspace[: max(len(controls) - 2, 0)])
                if len(cube_scratch) < len(controls) - 2:
                    raise BooleanCandidateError(
                        "mapping: cube requires clean workspace"
                    )
                flips = tuple(
                    Instruction("x", (wire,))
                    for i, wire in enumerate(controls)
                    if not satisfying[0] >> i & 1
                )
                mapped = (
                    *flips,
                    *_clean_mcx_instructions(controls, target, cube_scratch),
                    *flips,
                )
                if len(gates) + len(mapped) > min(max_operations, 256):
                    raise BooleanCandidateError(
                        "budget: Boolean mapped operations",
                        exhausted=max_operations < 256,
                    )
                gates.extend(mapped)
                used_workspace = tuple(dict.fromkeys((*used_workspace, *cube_scratch)))
                continue
        for mask in sorted(terms):
            controls = tuple(w for i, w in enumerate(wires) if mask & (1 << i))
            degree = len(controls)
            if degree > 3 or (degree == 3 and not workspace):
                raise BooleanCandidateError(
                    "mapping: requires more workspace or factoring"
                )
            if len(gates) + (3 if degree == 3 else 1) > min(max_operations, 256):
                raise BooleanCandidateError(
                    "budget: Boolean mapped operations", exhausted=max_operations < 256
                )
            if degree == 3:
                scratch = workspace[0]
                used_workspace = tuple(dict.fromkeys((*used_workspace, scratch)))
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
        analysis,
    )
