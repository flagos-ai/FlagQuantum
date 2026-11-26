"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.quafu_dynamic_legality import (
    _TASK_API_REQUEST_FIELDS as _TASK_API_REQUEST_FIELDS,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_legality import __all__ as __all__
from flagquantum.compute.remote.qpu.quafu_dynamic_legality import (
    _preview_quafu_dynamic_request as _preview_quafu_dynamic_request,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_legality import (
    _provider_contract_blockers as _provider_contract_blockers,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_legality import (
    _quafu_extension as _quafu_extension,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_legality import (
    _QuafuDynamicLegalityPreview as _QuafuDynamicLegalityPreview,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_legality import (
    _routing_blockers as _routing_blockers,
)
from flagquantum.compute.remote.qpu.quafu_dynamic_legality import (
    _snapshot_fact as _snapshot_fact,
)

_implementation = _import_module(
    "flagquantum.compute.remote.qpu.quafu_dynamic_legality"
)
_sys.modules[__name__] = _implementation
