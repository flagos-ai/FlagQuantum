"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.ionq import (
    _CONNECTIVITY_NAMES as _CONNECTIVITY_NAMES,
)
from flagquantum.compute.remote.qpu.ionq import (
    _IONQ_STATUS as _IONQ_STATUS,
)
from flagquantum.compute.remote.qpu.ionq import (
    _SIMULATOR_MARKERS as _SIMULATOR_MARKERS,
)
from flagquantum.compute.remote.qpu.ionq import (
    IONQ_API_BASE as IONQ_API_BASE,
)
from flagquantum.compute.remote.qpu.ionq import (
    IONQ_GATE_SETS as IONQ_GATE_SETS,
)
from flagquantum.compute.remote.qpu.ionq import (
    IONQ_PROVIDER as IONQ_PROVIDER,
)
from flagquantum.compute.remote.qpu.ionq import (
    IonQProvider as IonQProvider,
)
from flagquantum.compute.remote.qpu.ionq import __all__ as __all__
from flagquantum.compute.remote.qpu.ionq import (
    _ionq_outcome as _ionq_outcome,
)
from flagquantum.compute.remote.qpu.ionq import (
    _program_format as _program_format,
)
from flagquantum.compute.remote.qpu.ionq import (
    ionq_backend_profile as ionq_backend_profile,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.ionq")
_sys.modules[__name__] = _implementation
