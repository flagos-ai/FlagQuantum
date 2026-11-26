"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.chemistry.models.active_space import (
    ActiveSpaceResult as ActiveSpaceResult,
)
from flagquantum.algorithms.chemistry.models.active_space import (
    select_active_space as select_active_space,
)

_implementation = _import_module("flagquantum.algorithms.chemistry.models.active_space")
_sys.modules[__name__] = _implementation
