"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._program_submission import __all__ as __all__
from flagquantum.compute.remote.compute._program_submission import (
    _ProgramSubmissionMixin as _ProgramSubmissionMixin,
)
from flagquantum.compute.remote.compute._program_submission import (
    decode_job_result as decode_job_result,
)

_implementation = _import_module(
    "flagquantum.compute.remote.compute._program_submission"
)
_sys.modules[__name__] = _implementation
