"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._workspace_results import __all__ as __all__
from flagquantum.compute.remote.compute._workspace_results import (
    _decode_value as _decode_value,
)
from flagquantum.compute.remote.compute._workspace_results import (
    decode_tensor as decode_tensor,
)
from flagquantum.compute.remote.compute._workspace_results import (
    measurement_result as measurement_result,
)

_implementation = _import_module(
    "flagquantum.compute.remote.compute._workspace_results"
)
_sys.modules[__name__] = _implementation
