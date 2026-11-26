"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.planner.synthesis import (
    _UNBOUNDED_LIMITS as _UNBOUNDED_LIMITS,
)
from flagquantum.runtime.planner.synthesis import (
    CheckedExecution as CheckedExecution,
)
from flagquantum.runtime.planner.synthesis import (
    ExecutionContractError as ExecutionContractError,
)
from flagquantum.runtime.planner.synthesis import (
    _check_basis as _check_basis,
)
from flagquantum.runtime.planner.synthesis import (
    _check_limits as _check_limits,
)
from flagquantum.runtime.planner.synthesis import (
    _controls as _controls,
)
from flagquantum.runtime.planner.synthesis import (
    _inspect_execution_with_resources as _inspect_execution_with_resources,
)
from flagquantum.runtime.planner.synthesis import (
    _layout as _layout,
)
from flagquantum.runtime.planner.synthesis import (
    _placement as _placement,
)
from flagquantum.runtime.planner.synthesis import (
    _program as _program,
)
from flagquantum.runtime.planner.synthesis import (
    inspect_execution as inspect_execution,
)

_implementation = _import_module("flagquantum.runtime.planner.synthesis")
_sys.modules[__name__] = _implementation
