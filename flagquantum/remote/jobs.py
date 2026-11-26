"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.jobs import (
    JobStatus as JobStatus,
)
from flagquantum.compute.remote.jobs import (
    MultiTargetSubmission as MultiTargetSubmission,
)
from flagquantum.compute.remote.jobs import (
    RemoteJob as RemoteJob,
)
from flagquantum.compute.remote.jobs import (
    TargetSubmission as TargetSubmission,
)
from flagquantum.compute.remote.jobs import __all__ as __all__
from flagquantum.compute.remote.jobs import (
    _normalize as _normalize,
)
from flagquantum.compute.remote.jobs import (
    _prepare_quafu_package as _prepare_quafu_package,
)
from flagquantum.compute.remote.jobs import (
    _Receipt as _Receipt,
)
from flagquantum.compute.remote.jobs import (
    _submit_quafu as _submit_quafu,
)
from flagquantum.compute.remote.jobs import (
    restore_job as restore_job,
)
from flagquantum.compute.remote.jobs import (
    submit as submit,
)
from flagquantum.compute.remote.jobs import (
    submit_to_targets as submit_to_targets,
)

_implementation = _import_module("flagquantum.compute.remote.jobs")
_sys.modules[__name__] = _implementation
