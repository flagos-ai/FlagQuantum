"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.contracts import (
    DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA as DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA,
)
from flagquantum.compute.remote.qpu.contracts import (
    DeploymentResult as DeploymentResult,
)
from flagquantum.compute.remote.qpu.contracts import (
    ProviderSubmissionPreview as ProviderSubmissionPreview,
)
from flagquantum.compute.remote.qpu.contracts import (
    ProviderTaskHandle as ProviderTaskHandle,
)
from flagquantum.compute.remote.qpu.contracts import (
    QuantumProvider as QuantumProvider,
)
from flagquantum.compute.remote.qpu.contracts import __all__ as __all__
from flagquantum.compute.remote.qpu.contracts import (
    build_result_metadata as build_result_metadata,
)
from flagquantum.compute.remote.qpu.contracts import (
    build_submission_receipt as build_submission_receipt,
)
from flagquantum.compute.remote.qpu.contracts import (
    preflight_submission as preflight_submission,
)
from flagquantum.compute.remote.qpu.contracts import (
    validate_deployment_result as validate_deployment_result,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.contracts")
_sys.modules[__name__] = _implementation
