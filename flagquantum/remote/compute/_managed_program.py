"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.compute._managed_program import (
    _IGNORED_PARTS as _IGNORED_PARTS,
)
from flagquantum.compute.remote.compute._managed_program import (
    _MANAGED_ROOT as _MANAGED_ROOT,
)
from flagquantum.compute.remote.compute._managed_program import (
    _MAX_RECEIPT_BYTES as _MAX_RECEIPT_BYTES,
)
from flagquantum.compute.remote.compute._managed_program import (
    _MAX_RESULT_BYTES as _MAX_RESULT_BYTES,
)
from flagquantum.compute.remote.compute._managed_program import (
    _MAX_SOURCE_BYTES as _MAX_SOURCE_BYTES,
)
from flagquantum.compute.remote.compute._managed_program import (
    _REMOTE_PYTHON as _REMOTE_PYTHON,
)
from flagquantum.compute.remote.compute._managed_program import __all__ as __all__
from flagquantum.compute.remote.compute._managed_program import (
    _ensure_source as _ensure_source,
)
from flagquantum.compute.remote.compute._managed_program import (
    _run_ssh as _run_ssh,
)
from flagquantum.compute.remote.compute._managed_program import (
    _source_archive as _source_archive,
)
from flagquantum.compute.remote.compute._managed_program import (
    _source_files as _source_files,
)
from flagquantum.compute.remote.compute._managed_program import (
    _upload_archive as _upload_archive,
)
from flagquantum.compute.remote.compute._managed_program import (
    persist_managed_receipt as persist_managed_receipt,
)
from flagquantum.compute.remote.compute._managed_program import (
    read_managed_result as read_managed_result,
)
from flagquantum.compute.remote.compute._managed_program import (
    restore_managed_receipt as restore_managed_receipt,
)
from flagquantum.compute.remote.compute._managed_program import (
    submit_managed_program as submit_managed_program,
)

_implementation = _import_module("flagquantum.compute.remote.compute._managed_program")
_sys.modules[__name__] = _implementation
