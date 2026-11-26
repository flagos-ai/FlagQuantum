"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.training.module import (
    CircuitBuilder as CircuitBuilder,
)
from flagquantum.runtime.training.module import (
    ExecutionResult as ExecutionResult,
)
from flagquantum.runtime.training.module import (
    Module as Module,
)
from flagquantum.runtime.training.module import (
    RuntimePolicy as RuntimePolicy,
)
from flagquantum.runtime.training.module import __all__ as __all__
from flagquantum.runtime.training.module import (
    normalize_execution_result as normalize_execution_result,
)

_implementation = _import_module("flagquantum.runtime.training.module")
_sys.modules[__name__] = _implementation
