"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._native_job import (
    _LOG_QUERY_WINDOW_MS as _LOG_QUERY_WINDOW_MS,
)
from flagquantum.compute.remote.compute._native_job import (
    _MAX_COMMAND as _MAX_COMMAND,
)
from flagquantum.compute.remote.compute._native_job import (
    _MAX_RESULT as _MAX_RESULT,
)
from flagquantum.compute.remote.compute._native_job import (
    NativeJiudingJobClient as NativeJiudingJobClient,
)
from flagquantum.compute.remote.compute._native_job import (
    _command as _command,
)
from flagquantum.compute.remote.compute._native_job import (
    _decode_lines as _decode_lines,
)
from flagquantum.compute.remote.compute._native_job import (
    _one as _one,
)

_implementation = _import_module("flagquantum.compute.remote.compute._native_job")
_sys.modules[__name__] = _implementation
