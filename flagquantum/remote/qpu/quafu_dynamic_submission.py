"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.quafu_dynamic_submission import (
    _RESULT_BIT_ORDER as _RESULT_BIT_ORDER,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_submission import (
    _SDK_PROTOCOL as _SDK_PROTOCOL,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_submission import __all__ as __all__
from flagquantum.compute.remote.qpu.quafu_dynamic_submission import (
    _dynamic_receipt_identity as _dynamic_receipt_identity,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_submission import (
    _normalize_quafu_dynamic_mock_result as _normalize_quafu_dynamic_mock_result,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_submission import (
    _submit_quafu_dynamic_once as _submit_quafu_dynamic_once,
)

_implementation = _import_module(
    "flagquantum.compute.remote.qpu.quafu_dynamic_submission"
)
_sys.modules[__name__] = _implementation
