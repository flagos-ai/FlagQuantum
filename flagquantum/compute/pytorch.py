"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.local.pytorch import (
    CPUPlatformRuntime as CPUPlatformRuntime,
)
from flagquantum.compute.local.pytorch import (
    CUDAPlatformRuntime as CUDAPlatformRuntime,
)
from flagquantum.compute.local.pytorch import __all__ as __all__
from flagquantum.compute.local.pytorch import (
    _cgroup_memory_snapshot as _cgroup_memory_snapshot,
)
from flagquantum.compute.local.pytorch import (
    _CPUEvent as _CPUEvent,
)
from flagquantum.compute.local.pytorch import (
    _CUDAStreamFactory as _CUDAStreamFactory,
)
from flagquantum.compute.local.pytorch import (
    _darwin_available_memory_bytes as _darwin_available_memory_bytes,
)
from flagquantum.compute.local.pytorch import (
    _file_integer as _file_integer,
)
from flagquantum.compute.local.pytorch import (
    _host_memory_snapshot as _host_memory_snapshot,
)
from flagquantum.compute.local.pytorch import (
    _minimum_known as _minimum_known,
)
from flagquantum.compute.local.pytorch import (
    _sysconf_bytes as _sysconf_bytes,
)

_implementation = _import_module("flagquantum.compute.local.pytorch")
_sys.modules[__name__] = _implementation
