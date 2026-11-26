"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.quantinuum import (
    _QUANTINUUM_STATUS as _QUANTINUUM_STATUS,
)
from flagquantum.compute.remote.qpu.quantinuum import (
    _SIMULATOR_MARKERS as _SIMULATOR_MARKERS,
)
from flagquantum.compute.remote.qpu.quantinuum import (
    QUANTINUUM_API_BASE as QUANTINUUM_API_BASE,
)
from flagquantum.compute.remote.qpu.quantinuum import (
    QUANTINUUM_LANGUAGES as QUANTINUUM_LANGUAGES,
)
from flagquantum.compute.remote.qpu.quantinuum import (
    QUANTINUUM_PROVIDER as QUANTINUUM_PROVIDER,
)
from flagquantum.compute.remote.qpu.quantinuum import (
    QuantinuumProvider as QuantinuumProvider,
)
from flagquantum.compute.remote.qpu.quantinuum import __all__ as __all__
from flagquantum.compute.remote.qpu.quantinuum import (
    _language as _language,
)
from flagquantum.compute.remote.qpu.quantinuum import (
    _quantinuum_counts as _quantinuum_counts,
)
from flagquantum.compute.remote.qpu.quantinuum import (
    _quantinuum_outcome as _quantinuum_outcome,
)
from flagquantum.compute.remote.qpu.quantinuum import (
    quantinuum_backend_profile as quantinuum_backend_profile,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.quantinuum")
_sys.modules[__name__] = _implementation
