"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._job_payload import (
    experiment_body as experiment_body,
)

_implementation = _import_module("flagquantum.compute.remote.compute._job_payload")
_sys.modules[__name__] = _implementation
