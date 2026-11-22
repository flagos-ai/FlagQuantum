# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix id/* union definitions, Apache-2.0.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/id

"""Typed admission and raw-word projections for physical operand union contexts."""

from enum import Enum

from ._operand_tags import PAYLOAD_MASK
from .operand_identity import OperandIdentity, OperandKind


class OperandDomain(Enum):
    """Static union context retained separately from its scalar identity."""

    CLASSICAL_BOOLEAN = "classical_boolean"
    CLASSICAL_FALSE = "classical_false"
    QUANTUM_CLASSICAL = "quantum_classical"
    Z_OPERAND = "z_operand"
    QUANTUM_FALSE = "quantum_false"
    QUANTUM_MINUS = "quantum_minus"
    QUANTUM_TRUE = "quantum_true"
    X_OPERAND = "x_operand"
    MIXED_OPERAND = "mixed_operand"


_BOOLEAN = OperandKind.BOOLEAN
_CLASSICAL = OperandKind.CLASSICAL
_QUANTUM = OperandKind.QUANTUM
_X_BOOLEAN = OperandKind.X_BOOLEAN
_X_CLASSICAL = OperandKind.X_CLASSICAL
_ALLOWED = {
    OperandDomain.CLASSICAL_BOOLEAN: frozenset((_BOOLEAN, _CLASSICAL)),
    OperandDomain.CLASSICAL_FALSE: frozenset((_BOOLEAN, _CLASSICAL)),
    OperandDomain.QUANTUM_CLASSICAL: frozenset((_QUANTUM, _CLASSICAL)),
    OperandDomain.Z_OPERAND: frozenset((_BOOLEAN, _CLASSICAL, _QUANTUM)),
    OperandDomain.QUANTUM_FALSE: frozenset((_QUANTUM,)),
    OperandDomain.QUANTUM_MINUS: frozenset((_QUANTUM, _X_BOOLEAN)),
    OperandDomain.QUANTUM_TRUE: frozenset((_BOOLEAN, _QUANTUM)),
    OperandDomain.X_OPERAND: frozenset((_QUANTUM, _X_BOOLEAN, _X_CLASSICAL)),
    OperandDomain.MIXED_OPERAND: frozenset(
        (_BOOLEAN, _CLASSICAL, _QUANTUM, _X_BOOLEAN, _X_CLASSICAL)
    ),
}
_CONSTANTS = {
    OperandDomain.CLASSICAL_FALSE: OperandIdentity(_BOOLEAN, 0),
    OperandDomain.QUANTUM_MINUS: OperandIdentity(_X_BOOLEAN, 1),
    OperandDomain.QUANTUM_TRUE: OperandIdentity(_BOOLEAN, 1),
}
_CASTS = {
    OperandDomain.CLASSICAL_BOOLEAN: frozenset((_BOOLEAN, _CLASSICAL)),
    OperandDomain.CLASSICAL_FALSE: frozenset((_CLASSICAL,)),
    OperandDomain.QUANTUM_CLASSICAL: frozenset((_QUANTUM, _CLASSICAL)),
    OperandDomain.Z_OPERAND: frozenset((_BOOLEAN, _CLASSICAL, _QUANTUM)),
    OperandDomain.QUANTUM_FALSE: frozenset((_QUANTUM,)),
    OperandDomain.QUANTUM_MINUS: frozenset((_QUANTUM,)),
    OperandDomain.QUANTUM_TRUE: frozenset((_QUANTUM,)),
    OperandDomain.X_OPERAND: frozenset((_QUANTUM, _X_CLASSICAL)),
    OperandDomain.MIXED_OPERAND: frozenset(
        (_BOOLEAN, _CLASSICAL, _QUANTUM, _X_BOOLEAN, _X_CLASSICAL)
    ),
}


def admit_operand_identity(
    identity: OperandIdentity, domain: OperandDomain
) -> OperandIdentity:
    """Check the source union constructor alternatives without losing identity."""
    _validate_domain(domain)
    if not isinstance(identity, OperandIdentity):
        raise ValueError("Operand union admission requires a typed identity")
    if identity.kind not in _ALLOWED[domain]:
        raise ValueError(f"Operand family is outside {domain.value}")
    constant = _CONSTANTS.get(domain)
    if constant is not None and identity.kind is constant.kind and identity != constant:
        raise ValueError(f"Operand constant is outside {domain.value}")
    return identity


def default_operand_identity(domain: OperandDomain) -> OperandIdentity:
    """Retain the original default word, including its untyped false sentinel."""
    _validate_domain(domain)
    if domain in (OperandDomain.QUANTUM_MINUS, OperandDomain.QUANTUM_TRUE):
        raise ValueError(f"Operand domain {domain.value} has no default constructor")
    if domain is OperandDomain.X_OPERAND:
        return OperandIdentity(_X_BOOLEAN, 0)
    return OperandIdentity(_BOOLEAN, 0)


def _validate_domain(domain: OperandDomain) -> None:
    if not isinstance(domain, OperandDomain):
        raise ValueError("Physical operand domain must be an OperandDomain")


def _validate_word(word: int) -> None:
    if type(word) is not int or not 0 <= word < (1 << 32):
        raise ValueError("Physical operand word must be an unsigned uint32 integer")


def cast_operand_word(
    word: int, domain: OperandDomain, target: OperandKind
) -> OperandIdentity:
    """Apply source explicit casts, including their unchecked payload projection.

    A mixed-domain Boolean cast selects only exact true. The narrower Z-domain
    casts use full-word truth. X Boolean conversion selects only exact minus.
    """
    _validate_domain(domain)
    _validate_word(word)
    if target not in _CASTS[domain]:
        raise ValueError(f"Operand cast is unavailable in {domain.value}")
    if target is _BOOLEAN:
        value = (
            int(word == 1) if domain is OperandDomain.MIXED_OPERAND else int(bool(word))
        )
    elif target is _X_BOOLEAN:
        value = int(word == (3 << 29) | 1)
    else:
        value = word & PAYLOAD_MASK
    return OperandIdentity(target, value)


def format_operand_word(word: int, domain: OperandDomain) -> str:
    """Render a union diagnostic, preserving original mutated-word fallbacks."""
    _validate_domain(domain)
    _validate_word(word)
    tag, payload = word >> 29, word & PAYLOAD_MASK
    if domain is OperandDomain.QUANTUM_CLASSICAL:
        return f"{'q' if tag == 2 else 'b'}{payload}"
    if domain in (OperandDomain.CLASSICAL_FALSE, OperandDomain.QUANTUM_FALSE):
        expected = 1 if domain is OperandDomain.CLASSICAL_FALSE else 2
        return f"{'b' if tag == 1 else 'q'}{payload}" if tag == expected else "False"
    if domain is OperandDomain.QUANTUM_TRUE:
        raise ValueError("Quantum/true union has no source diagnostic renderer")
    if domain is OperandDomain.QUANTUM_MINUS:
        if tag == 2:
            return f"q{payload}"
        if word == (3 << 29) | 1:
            return "|->"
        return "{invalid quantum/minus operand instance}"
    if domain is OperandDomain.CLASSICAL_BOOLEAN:
        return f"b{payload}" if tag == 1 else str(bool(word)).lower()
    if domain is OperandDomain.Z_OPERAND:
        if tag in (1, 2):
            return f"{'b' if tag == 1 else 'q'}{payload}"
        return str(bool(word))
    if domain is OperandDomain.X_OPERAND:
        if tag == 2:
            return f"q{payload}"
        if tag == 4:
            return f"(X)b{payload}"
        if word in (3 << 29, (3 << 29) | 1):
            return "|->" if payload else "|+>"
        return "XOperand{???}"
    if tag in (1, 2, 4):
        prefix = {1: "b", 2: "q", 4: "xb"}[tag]
        return f"{prefix}{payload}"
    if tag == 3:
        return f"XBool({str(word == (3 << 29) | 1).lower()})"
    if tag == 0:
        return str(word == 1).lower()
    return f"MixedOperand{{.physical_word={word}}}"
