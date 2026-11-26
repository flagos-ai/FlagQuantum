"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.kaiwu.contracts import (
    KAIWU_TASK_RECEIPT_SCHEMA as KAIWU_TASK_RECEIPT_SCHEMA,
)
from flagquantum.compute.remote.kaiwu.contracts import (
    FrozenIsingMatrix as FrozenIsingMatrix,
)
from flagquantum.compute.remote.kaiwu.contracts import (
    KaiwuJobStatus as KaiwuJobStatus,
)
from flagquantum.compute.remote.kaiwu.contracts import (
    KaiwuTaskClient as KaiwuTaskClient,
)
from flagquantum.compute.remote.kaiwu.contracts import (
    KaiwuTaskMode as KaiwuTaskMode,
)
from flagquantum.compute.remote.kaiwu.contracts import (
    KaiwuTaskReceipt as KaiwuTaskReceipt,
)
from flagquantum.compute.remote.kaiwu.contracts import (
    KaiwuTaskResult as KaiwuTaskResult,
)
from flagquantum.compute.remote.kaiwu.contracts import (
    MatrixInput as MatrixInput,
)
from flagquantum.compute.remote.kaiwu.contracts import __all__ as __all__

_implementation = _import_module("flagquantum.compute.remote.kaiwu.contracts")
_sys.modules[__name__] = _implementation
