"""Historical import facade for the relocated implementation."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

from flagquantum.compute.remote.qpu import (
    DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA as DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA,
)
from flagquantum.compute.remote.qpu import (
    IONQ_GATE_SETS as IONQ_GATE_SETS,
)
from flagquantum.compute.remote.qpu import (
    AmazonBraketProvider as AmazonBraketProvider,
)
from flagquantum.compute.remote.qpu import (
    AzureQuantumProvider as AzureQuantumProvider,
)
from flagquantum.compute.remote.qpu import (
    AzureSubmissionPreview as AzureSubmissionPreview,
)
from flagquantum.compute.remote.qpu import (
    BraketSubmissionPreview as BraketSubmissionPreview,
)
from flagquantum.compute.remote.qpu import (
    DeploymentResult as DeploymentResult,
)
from flagquantum.compute.remote.qpu import (
    HttpQuantumProvider as HttpQuantumProvider,
)
from flagquantum.compute.remote.qpu import (
    IonQProvider as IonQProvider,
)
from flagquantum.compute.remote.qpu import (
    IQMProvider as IQMProvider,
)
from flagquantum.compute.remote.qpu import (
    ProviderCredentials as ProviderCredentials,
)
from flagquantum.compute.remote.qpu import (
    ProviderEndpoints as ProviderEndpoints,
)
from flagquantum.compute.remote.qpu import (
    ProviderSubmissionPreview as ProviderSubmissionPreview,
)
from flagquantum.compute.remote.qpu import (
    ProviderTaskHandle as ProviderTaskHandle,
)
from flagquantum.compute.remote.qpu import (
    QuafuProvider as QuafuProvider,
)
from flagquantum.compute.remote.qpu import (
    QuantinuumProvider as QuantinuumProvider,
)
from flagquantum.compute.remote.qpu import (
    QuantumCloudTransport as QuantumCloudTransport,
)
from flagquantum.compute.remote.qpu import (
    QuantumProvider as QuantumProvider,
)
from flagquantum.compute.remote.qpu import (
    UrllibTransport as UrllibTransport,
)
from flagquantum.compute.remote.qpu import __all__ as __all__
from flagquantum.compute.remote.qpu import (
    azure_backend_profile as azure_backend_profile,
)
from flagquantum.compute.remote.qpu import (
    braket_backend_profile as braket_backend_profile,
)
from flagquantum.compute.remote.qpu import (
    build_result_metadata as build_result_metadata,
)
from flagquantum.compute.remote.qpu import (
    build_submission_receipt as build_submission_receipt,
)
from flagquantum.compute.remote.qpu import (
    ionq_backend_profile as ionq_backend_profile,
)
from flagquantum.compute.remote.qpu import (
    iqm_backend_profile as iqm_backend_profile,
)
from flagquantum.compute.remote.qpu import (
    preflight_submission as preflight_submission,
)
from flagquantum.compute.remote.qpu import (
    quafu_noise_model_from_chip_info as quafu_noise_model_from_chip_info,
)
from flagquantum.compute.remote.qpu import (
    quantinuum_backend_profile as quantinuum_backend_profile,
)
from flagquantum.compute.remote.qpu import (
    validate_deployment_result as validate_deployment_result,
)

_implementation = _import_module("flagquantum.compute.remote.qpu")
__doc__ = _implementation.__doc__ or ""


def __getattr__(name: str) -> _Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)
