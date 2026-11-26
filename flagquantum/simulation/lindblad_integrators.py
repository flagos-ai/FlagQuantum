"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.simulation.dynamics.integrators import (
    _INTEGRATOR_ORDERS as _INTEGRATOR_ORDERS,
)
from flagquantum.simulation.dynamics.integrators import (
    _SUBSTEPS as _SUBSTEPS,
)
from flagquantum.simulation.dynamics.integrators import (
    CRANK_NICOLSON as CRANK_NICOLSON,
)
from flagquantum.simulation.dynamics.integrators import (
    DEFAULT_INTEGRATOR as DEFAULT_INTEGRATOR,
)
from flagquantum.simulation.dynamics.integrators import (
    INTEGRATOR_NAMES as INTEGRATOR_NAMES,
)
from flagquantum.simulation.dynamics.integrators import (
    KRYLOV_EXPONENTIAL as KRYLOV_EXPONENTIAL,
)
from flagquantum.simulation.dynamics.integrators import (
    KRYLOV_SCHEMES as KRYLOV_SCHEMES,
)
from flagquantum.simulation.dynamics.integrators import (
    RUNGE_KUTTA as RUNGE_KUTTA,
)
from flagquantum.simulation.dynamics.integrators import (
    SOLVE_RESTART as SOLVE_RESTART,
)
from flagquantum.simulation.dynamics.integrators import (
    SUBSTEPS_PER_INTERVAL as SUBSTEPS_PER_INTERVAL,
)
from flagquantum.simulation.dynamics.integrators import __all__ as __all__
from flagquantum.simulation.dynamics.integrators import (
    _crank_nicolson_interval as _crank_nicolson_interval,
)
from flagquantum.simulation.dynamics.integrators import (
    _crank_nicolson_step as _crank_nicolson_step,
)
from flagquantum.simulation.dynamics.integrators import (
    _krylov_exponential_interval as _krylov_exponential_interval,
)
from flagquantum.simulation.dynamics.integrators import (
    _runge_kutta_4_step as _runge_kutta_4_step,
)
from flagquantum.simulation.dynamics.integrators import (
    advance_interval as advance_interval,
)
from flagquantum.simulation.dynamics.integrators import (
    integrator_order as integrator_order,
)
from flagquantum.simulation.dynamics.integrators import (
    substeps_per_interval as substeps_per_interval,
)

_implementation = _import_module("flagquantum.simulation.dynamics.integrators")
_sys.modules[__name__] = _implementation
