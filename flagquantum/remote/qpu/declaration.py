"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.declaration import (
    _MISSING as _MISSING,
)
from flagquantum.compute.remote.qpu.declaration import __all__ as __all__
from flagquantum.compute.remote.qpu.declaration import (
    connectivity_edges as connectivity_edges,
)
from flagquantum.compute.remote.qpu.declaration import (
    declared_count as declared_count,
)
from flagquantum.compute.remote.qpu.declaration import (
    declared_flag as declared_flag,
)
from flagquantum.compute.remote.qpu.declaration import (
    declared_gate_names as declared_gate_names,
)
from flagquantum.compute.remote.qpu.declaration import (
    declared_text as declared_text,
)
from flagquantum.compute.remote.qpu.declaration import (
    declared_value as declared_value,
)
from flagquantum.compute.remote.qpu.declaration import (
    declared_width as declared_width,
)
from flagquantum.compute.remote.qpu.declaration import (
    edge_pair as edge_pair,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.declaration")
_sys.modules[__name__] = _implementation
