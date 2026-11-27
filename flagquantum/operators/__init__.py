"""Stable operator discovery, registration, and superoperator interfaces."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from flagquantum.core.operators.registry import OperatorRegistry as OperatorRegistry
from flagquantum.core.operators.schema import GateInfo as GateInfo
from flagquantum.core.operators.schema import gate_info as gate_info
from flagquantum.core.operators.superoperator import (
    DEFAULT_DENSE_MATRIX_BYTES as DEFAULT_DENSE_MATRIX_BYTES,
)
from flagquantum.core.operators.superoperator import SuperOperator as SuperOperator

if TYPE_CHECKING:
    from flagquantum.compiler.construction.operator_registration import (
        register_operator as register_operator,
    )

__all__ = (
    "DEFAULT_DENSE_MATRIX_BYTES",
    "GateInfo",
    "OperatorRegistry",
    "SuperOperator",
    "gate_info",
    "register_operator",
)


def __getattr__(name: str) -> Any:
    if name == "register_operator":
        return getattr(
            import_module("flagquantum.compiler.construction.operator_registration"),
            name,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
