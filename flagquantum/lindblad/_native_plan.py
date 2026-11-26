"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.evolution._native_plan import (
    NATIVE_VERSION as NATIVE_VERSION,
)
from flagquantum.runtime.evolution._native_plan import (
    _numeric_values as _numeric_values,
)
from flagquantum.runtime.evolution._native_plan import (
    build_native_plan as build_native_plan,
)
from flagquantum.runtime.evolution._native_plan import (
    native_request as native_request,
)
from flagquantum.runtime.evolution._native_plan import (
    validate_native_plan as validate_native_plan,
)

_implementation = _import_module("flagquantum.runtime.evolution._native_plan")
_sys.modules[__name__] = _implementation
