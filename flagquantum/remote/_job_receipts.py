"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote._job_receipts import (
    _SCHEMA as _SCHEMA,
)
from flagquantum.compute.remote._job_receipts import (
    _SHA256 as _SHA256,
)
from flagquantum.compute.remote._job_receipts import (
    _complete as _complete,
)
from flagquantum.compute.remote._job_receipts import (
    _Journal as _Journal,
)
from flagquantum.compute.remote._job_receipts import (
    _read_journal as _read_journal,
)
from flagquantum.compute.remote._job_receipts import (
    _receipt_digest as _receipt_digest,
)
from flagquantum.compute.remote._job_receipts import (
    _reserve as _reserve,
)
from flagquantum.compute.remote._job_receipts import (
    _SubmissionSettings as _SubmissionSettings,
)
from flagquantum.compute.remote._job_receipts import (
    request_digest as request_digest,
)
from flagquantum.compute.remote._job_receipts import (
    resume_or_submit as resume_or_submit,
)

_implementation = _import_module("flagquantum.compute.remote._job_receipts")
_sys.modules[__name__] = _implementation
