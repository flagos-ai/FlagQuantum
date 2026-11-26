"""Historical registry and frontend registration imports."""

# Keep the historical namespace available to get_type_hints and serializers.
from collections.abc import Mapping as Mapping
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from dataclasses import replace as replace
from importlib import import_module
from types import MappingProxyType as MappingProxyType
from typing import TYPE_CHECKING, Any

from flagquantum.core.operators.registry import OperatorRegistry as OperatorRegistry

from ..core.operator_schema import ADJOINT_RULES as ADJOINT_RULES
from ..core.operator_schema import OPERATOR_ALIASES as OPERATOR_ALIASES
from ..core.operator_schema import OPERATOR_SCHEMAS as OPERATOR_SCHEMAS
from ..core.operator_schema import OperatorSchema as OperatorSchema

if TYPE_CHECKING:
    from flagquantum.compiler.construction.operator_registration import (
        _install_circuit_methods as _install_circuit_methods,
    )
    from flagquantum.compiler.construction.operator_registration import (
        _spelling_surface as _spelling_surface,
    )
    from flagquantum.compiler.construction.operator_registration import (
        register_operator as register_operator,
    )

__all__ = ["OperatorRegistry", "register_operator"]


def __getattr__(name: str) -> Any:
    if name in {"register_operator", "_install_circuit_methods", "_spelling_surface"}:
        return getattr(
            import_module("flagquantum.compiler.construction.operator_registration"),
            name,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
