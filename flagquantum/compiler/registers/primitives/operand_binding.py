"""Normalize typed operands for the existing register emitters."""

from ...artifacts.operand_identity import OperandIdentity, OperandKind
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ._mixed_gates import _Control, _XBasisTarget

_Operand = _Control | _XBasisTarget


def bind_operand(value: object) -> _Operand:
    """Resolve one admitted Z-, X- or quantum operand without changing its slot."""
    if isinstance(value, OperandIdentity):
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
            case OperandKind.REGISTER:
                raise ValueError("A register identity is not a physical operand")
    if isinstance(value, (ClassicalFieldCoefficient, _XBasisTarget, bool)):
        return value
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value in ("|+>", "|->"):
        return _XBasisTarget(value == "|->")
    raise ValueError("Expected a quantum wire, classical bit or basis literal")
