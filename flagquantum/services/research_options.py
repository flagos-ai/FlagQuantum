"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._research_options import (
    ResearchExecutionOptions as ResearchExecutionOptions,
)
from flagquantum.algorithms.optimization._research_options import __all__ as __all__
from flagquantum.algorithms.optimization._research_options import (
    validate_research_execution_options as validate_research_execution_options,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._research_options"
)
_sys.modules[__name__] = _implementation
