"""Historical import facade for the relocated implementation."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

from flagquantum.runtime.evolution import (
    INTEGRATOR_NAMES as INTEGRATOR_NAMES,
)
from flagquantum.runtime.evolution import (
    CollapseOperator as CollapseOperator,
)
from flagquantum.runtime.evolution import (
    EvolutionResult as EvolutionResult,
)
from flagquantum.runtime.evolution import (
    LindbladPlan as LindbladPlan,
)
from flagquantum.runtime.evolution import __all__ as __all__
from flagquantum.runtime.evolution import (
    adjoint_gradient as adjoint_gradient,
)
from flagquantum.runtime.evolution import (
    amplitude_damping as amplitude_damping,
)
from flagquantum.runtime.evolution import (
    plan as plan,
)
from flagquantum.runtime.evolution import (
    run as run,
)

_implementation = _import_module("flagquantum.runtime.evolution")
__doc__ = _implementation.__doc__ or ""


def __getattr__(name: str) -> _Any:
    if name.startswith("_"):
        raise AttributeError(name)
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)
