"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._program_job import (
    _MAX_REQUEST_BYTES as _MAX_REQUEST_BYTES,
)
from flagquantum.compute.remote.compute._program_job import (
    PROGRAM_JOB_SCHEMA as PROGRAM_JOB_SCHEMA,
)
from flagquantum.compute.remote.compute._program_job import (
    PROGRAM_JOB_VERSION as PROGRAM_JOB_VERSION,
)
from flagquantum.compute.remote.compute._program_job import __all__ as __all__
from flagquantum.compute.remote.compute._program_job import (
    decode_program_result as decode_program_result,
)
from flagquantum.compute.remote.compute._program_job import (
    execute_request as execute_request,
)
from flagquantum.compute.remote.compute._program_job import (
    write_program_bundle as write_program_bundle,
)

_implementation = _import_module("flagquantum.compute.remote.compute._program_job")
_sys.modules[__name__] = _implementation
