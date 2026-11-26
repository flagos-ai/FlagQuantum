"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute.jiuding import (
    _MAX_BATCH_ITEMS as _MAX_BATCH_ITEMS,
)
from flagquantum.compute.remote.compute.jiuding import (
    _SUPPORTED_RESIDENT_MEASUREMENTS as _SUPPORTED_RESIDENT_MEASUREMENTS,
)
from flagquantum.compute.remote.compute.jiuding import (
    JiudingClient as JiudingClient,
)
from flagquantum.compute.remote.compute.jiuding import (
    _measurement_program as _measurement_program,
)
from flagquantum.compute.remote.compute.jiuding import (
    _NoRedirect as _NoRedirect,
)

_implementation = _import_module("flagquantum.compute.remote.compute.jiuding")
_sys.modules[__name__] = _implementation
