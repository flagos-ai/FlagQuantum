"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.iqm import (
    _CONNECTIVITY_NAMES as _CONNECTIVITY_NAMES,
)
from flagquantum.compute.remote.qpu.iqm import (
    _DEVICE_TYPE_NAMES as _DEVICE_TYPE_NAMES,
)
from flagquantum.compute.remote.qpu.iqm import (
    _IQM_STATUS as _IQM_STATUS,
)
from flagquantum.compute.remote.qpu.iqm import (
    _SIMULATOR_TYPES as _SIMULATOR_TYPES,
)
from flagquantum.compute.remote.qpu.iqm import (
    IQM_API_BASE as IQM_API_BASE,
)
from flagquantum.compute.remote.qpu.iqm import (
    IQM_NATIVE_GATES as IQM_NATIVE_GATES,
)
from flagquantum.compute.remote.qpu.iqm import (
    IQM_PROVIDER as IQM_PROVIDER,
)
from flagquantum.compute.remote.qpu.iqm import (
    IQMProvider as IQMProvider,
)
from flagquantum.compute.remote.qpu.iqm import __all__ as __all__
from flagquantum.compute.remote.qpu.iqm import (
    _iqm_outcome as _iqm_outcome,
)
from flagquantum.compute.remote.qpu.iqm import (
    iqm_backend_profile as iqm_backend_profile,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.iqm")
_sys.modules[__name__] = _implementation
