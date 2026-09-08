"""Compute targets reached through an external task control plane.

Remote adapters submit work and translate external task identifiers, status,
and results. Directly controlled CPU, GPU, and accelerator resources belong in
``flagquantum.compute``.
"""

from .braket import (
    AmazonBraketProvider,
    BraketSubmissionPreview,
    braket_backend_profile,
)
from .contracts import (
    DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA,
    DeploymentResult,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_result,
)
from .http import (
    HttpQuantumProvider,
    ProviderCredentials,
    ProviderEndpoints,
    QuantumCloudTransport,
    UrllibTransport,
)
from .quafu import QuafuProvider
from .quafu_calibration import quafu_noise_model_from_chip_info

__all__ = (
    "AmazonBraketProvider",
    "BraketSubmissionPreview",
    "DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA",
    "DeploymentResult",
    "HttpQuantumProvider",
    "ProviderCredentials",
    "ProviderEndpoints",
    "ProviderTaskHandle",
    "QuafuProvider",
    "QuantumCloudTransport",
    "QuantumProvider",
    "UrllibTransport",
    "braket_backend_profile",
    "build_result_metadata",
    "build_submission_receipt",
    "quafu_noise_model_from_chip_info",
    "validate_deployment_result",
)
