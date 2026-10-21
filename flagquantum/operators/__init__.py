"""Stable operator schema, discovery interface, the registration table, and the
superoperator algebra."""

from ..core.operator_schema import GateInfo, gate_info
from .registry import OperatorRegistry, register_operator
from .superoperator import DEFAULT_DENSE_MATRIX_BYTES, SuperOperator

__all__ = (
    "DEFAULT_DENSE_MATRIX_BYTES",
    "GateInfo",
    "OperatorRegistry",
    "SuperOperator",
    "gate_info",
    "register_operator",
)
