"""Stable operator schema, discovery interface, and the superoperator algebra."""

from ..core.operator_schema import GateInfo, gate_info
from .superoperator import DEFAULT_DENSE_MATRIX_BYTES, SuperOperator

__all__ = (
    "DEFAULT_DENSE_MATRIX_BYTES",
    "GateInfo",
    "SuperOperator",
    "gate_info",
)
