"""Historical numerical import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.observability.mps import (
    MPS_PROFILE_PHASES as MPS_PROFILE_PHASES,
)
from flagquantum.runtime.observability.mps import (
    MPS_PROFILE_SCHEMA as MPS_PROFILE_SCHEMA,
)
from flagquantum.runtime.observability.mps import (
    MPS_TRACE_COMMUNICATION_LABELS as MPS_TRACE_COMMUNICATION_LABELS,
)
from flagquantum.runtime.observability.mps import (
    MPS_TRACE_OPERATION_LABELS as MPS_TRACE_OPERATION_LABELS,
)
from flagquantum.runtime.observability.mps import (
    build_mps_critical_path_report as build_mps_critical_path_report,
)
from flagquantum.runtime.observability.mps import (
    workload_fingerprint as workload_fingerprint,
)

_sys.modules[__name__] = _import_module("flagquantum.runtime.observability.mps")
