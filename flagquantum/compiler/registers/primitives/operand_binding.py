"""Normalize typed operands for the existing register emitters."""

from ...artifacts._operand_views import OperandDomain, admit_operand_identity
from ...artifacts.operand_identity import OperandIdentity, OperandKind
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ._mixed_gates import _Control, _XBasisTarget

_Operand = _Control | _XBasisTarget


def bind_operand(value: object) -> _Operand:
    """Resolve one admitted Z-, X- or quantum operand without changing its slot."""
    if isinstance(value, OperandIdentity):
        if value.kind is OperandKind.REGISTER:
            raise ValueError("A register identity is not a physical operand")
        value = admit_operand_identity(value, OperandDomain.MIXED_OPERAND)
        match value.kind:
            case OperandKind.BOOLEAN:
                return bool(value)
            case OperandKind.CLASSICAL:
                return ClassicalFieldCoefficient(value.value)
            case OperandKind.QUANTUM:
                return value.value
            case OperandKind.X_BOOLEAN:
                return _XBasisTarget(bool(value))
            case OperandKind.X_CLASSICAL:
                return _XBasisTarget(ClassicalFieldCoefficient(value.value))
    if isinstance(value, (ClassicalFieldCoefficient, _XBasisTarget, bool)):
        return value
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value in ("|+>", "|->"):
        return _XBasisTarget(value == "|->")
    raise ValueError("Expected a quantum wire, classical bit or basis literal")


def identify_operand(operand: _Operand) -> OperandIdentity:
    """Retain the family tag when storing a resolved physical operand."""
    if isinstance(operand, _XBasisTarget):
        if isinstance(operand.outcome, ClassicalFieldCoefficient):
            return OperandIdentity(OperandKind.X_CLASSICAL, operand.outcome.slot)
        return OperandIdentity(OperandKind.X_BOOLEAN, int(operand.outcome))
    if isinstance(operand, ClassicalFieldCoefficient):
        return OperandIdentity(OperandKind.CLASSICAL, operand.slot)
    if isinstance(operand, bool):
        return OperandIdentity(OperandKind.BOOLEAN, int(operand))
    return OperandIdentity(OperandKind.QUANTUM, operand)
