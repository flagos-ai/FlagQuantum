"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._job_results import (
    read_job_result as read_job_result,
)
from flagquantum.compute.remote.compute._job_results import (
    save_receipt as save_receipt,
)

_implementation = _import_module("flagquantum.compute.remote.compute._job_results")
_sys.modules[__name__] = _implementation
