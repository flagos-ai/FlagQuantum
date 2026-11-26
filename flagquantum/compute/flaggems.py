"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.local.flaggems import (
    _BACKEND_PROBE_FAILURES as _BACKEND_PROBE_FAILURES,
)
from flagquantum.compute.local.flaggems import (
    FLAGGEMS_EXPERIMENTAL_OPS as FLAGGEMS_EXPERIMENTAL_OPS,
)
from flagquantum.compute.local.flaggems import (
    FLAGGEMS_FEATURE_REQUESTS as FLAGGEMS_FEATURE_REQUESTS,
)
from flagquantum.compute.local.flaggems import (
    FLAGGEMS_NATIVE_PYTORCH_HOTSPOTS as FLAGGEMS_NATIVE_PYTORCH_HOTSPOTS,
)
from flagquantum.compute.local.flaggems import (
    FLAGGEMS_NON_OPERATOR_REQUIREMENTS as FLAGGEMS_NON_OPERATOR_REQUIREMENTS,
)
from flagquantum.compute.local.flaggems import (
    FLAGGEMS_OP_ALIASES as FLAGGEMS_OP_ALIASES,
)
from flagquantum.compute.local.flaggems import (
    FLAGGEMS_SAFE_OPS as FLAGGEMS_SAFE_OPS,
)
from flagquantum.compute.local.flaggems import (
    OperatorBackendAvailability as OperatorBackendAvailability,
)
from flagquantum.compute.local.flaggems import (
    OperatorBackendSession as OperatorBackendSession,
)
from flagquantum.compute.local.flaggems import (
    OperatorReplacementPlan as OperatorReplacementPlan,
)
from flagquantum.compute.local.flaggems import (
    OperatorValidationResult as OperatorValidationResult,
)
from flagquantum.compute.local.flaggems import __all__ as __all__
from flagquantum.compute.local.flaggems import (
    _active_flaggems_registered_keys as _active_flaggems_registered_keys,
)
from flagquantum.compute.local.flaggems import (
    _availability_from_module as _availability_from_module,
)
from flagquantum.compute.local.flaggems import (
    _env_bool as _env_bool,
)
from flagquantum.compute.local.flaggems import (
    _extract_flaggems_catalog_keys as _extract_flaggems_catalog_keys,
)
from flagquantum.compute.local.flaggems import (
    _import_flag_gems as _import_flag_gems,
)
from flagquantum.compute.local.flaggems import (
    _known_flaggems_catalog as _known_flaggems_catalog,
)
from flagquantum.compute.local.flaggems import (
    _normalize_backend_name as _normalize_backend_name,
)
from flagquantum.compute.local.flaggems import (
    _probe_flaggems_in_subprocess as _probe_flaggems_in_subprocess,
)
from flagquantum.compute.local.flaggems import (
    _real_objective as _real_objective,
)
from flagquantum.compute.local.flaggems import (
    _run_flaggems_op_smoke as _run_flaggems_op_smoke,
)
from flagquantum.compute.local.flaggems import (
    _split_ops as _split_ops,
)
from flagquantum.compute.local.flaggems import (
    _torch_dtype_from_string as _torch_dtype_from_string,
)
from flagquantum.compute.local.flaggems import (
    flaggems_availability as flaggems_availability,
)
from flagquantum.compute.local.flaggems import (
    flaggems_preflight as flaggems_preflight,
)
from flagquantum.compute.local.flaggems import (
    operator_backend as operator_backend,
)
from flagquantum.compute.local.flaggems import (
    operator_backend_from_env as operator_backend_from_env,
)
from flagquantum.compute.local.flaggems import (
    plan_operator_replacements as plan_operator_replacements,
)
from flagquantum.compute.local.flaggems import (
    validate_flaggems_ops as validate_flaggems_ops,
)

_implementation = _import_module("flagquantum.compute.local.flaggems")
_sys.modules[__name__] = _implementation
