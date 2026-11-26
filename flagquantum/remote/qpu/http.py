"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.http import (
    HttpQuantumProvider as HttpQuantumProvider,
)
from flagquantum.compute.remote.qpu.http import (
    ProviderCredentials as ProviderCredentials,
)
from flagquantum.compute.remote.qpu.http import (
    ProviderEndpoints as ProviderEndpoints,
)
from flagquantum.compute.remote.qpu.http import (
    QuantumCloudTransport as QuantumCloudTransport,
)
from flagquantum.compute.remote.qpu.http import (
    UrllibTransport as UrllibTransport,
)
from flagquantum.compute.remote.qpu.http import __all__ as __all__

_implementation = _import_module("flagquantum.compute.remote.qpu.http")
_sys.modules[__name__] = _implementation
