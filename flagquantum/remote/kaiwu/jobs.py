"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.kaiwu.jobs import (
    _MAX_PRIVATE_JSON_BYTES as _MAX_PRIVATE_JSON_BYTES,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _MAX_RECEIPT_CLOCK_SKEW as _MAX_RECEIPT_CLOCK_SKEW,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _PRIVATE_JSON_READ_CHUNK_BYTES as _PRIVATE_JSON_READ_CHUNK_BYTES,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    KaiwuRemoteJob as KaiwuRemoteJob,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _effective_uid as _effective_uid,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _freeze_matrix as _freeze_matrix,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _matrix_sha256 as _matrix_sha256,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _normalize_status as _normalize_status,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _open_private_directory as _open_private_directory,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _read_bounded_bytes as _read_bounded_bytes,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _read_private_json as _read_private_json,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _recompute_energy as _recompute_energy,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _reject_duplicate_json_keys as _reject_duplicate_json_keys,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _reject_nonfinite_json_constant as _reject_nonfinite_json_constant,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _validate_private_directory as _validate_private_directory,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _validate_receipt as _validate_receipt,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _validate_result as _validate_result,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _verify_open_directory_binding as _verify_open_directory_binding,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    _write_private_json_exclusive as _write_private_json_exclusive,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    new_receipt as new_receipt,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    restore_kaiwu_job as restore_kaiwu_job,
)
from flagquantum.compute.remote.kaiwu.jobs import (
    submit_kaiwu_task as submit_kaiwu_task,
)

_implementation = _import_module("flagquantum.compute.remote.kaiwu.jobs")
_sys.modules[__name__] = _implementation
