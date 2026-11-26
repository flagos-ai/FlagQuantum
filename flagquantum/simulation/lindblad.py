"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.dynamics.lindblad import (
    CollapseOperator as CollapseOperator,
)
from flagquantum.simulation.dynamics.lindblad import (
    EvolutionPlan as EvolutionPlan,
)
from flagquantum.simulation.dynamics.lindblad import (
    EvolutionResult as EvolutionResult,
)
from flagquantum.simulation.dynamics.lindblad import (
    EvolutionValidationError as EvolutionValidationError,
)
from flagquantum.simulation.dynamics.lindblad import __all__ as __all__
from flagquantum.simulation.dynamics.lindblad import (
    _complex_tensor_to_json as _complex_tensor_to_json,
)
from flagquantum.simulation.dynamics.lindblad import (
    _evolve_density_matrix as _evolve_density_matrix,
)
from flagquantum.simulation.dynamics.lindblad import (
    _plan_density_matrix_evolution as _plan_density_matrix_evolution,
)
from flagquantum.simulation.dynamics.lindblad import (
    amplitude_damping as amplitude_damping,
)
from flagquantum.simulation.dynamics.lindblad import (
    evolve_density_matrix as evolve_density_matrix,
)
from flagquantum.simulation.dynamics.lindblad import (
    plan_density_matrix_evolution as plan_density_matrix_evolution,
)

_implementation = _import_module("flagquantum.simulation.dynamics.lindblad")
_sys.modules[__name__] = _implementation
