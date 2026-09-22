"""Remote QPU adapters and their quantum task contracts."""

from .azure import (
    AzureQuantumProvider,
    AzureSubmissionPreview,
    azure_backend_profile,
)
from .braket import (
    AmazonBraketProvider,
    BraketSubmissionPreview,
    braket_backend_profile,
)
from .calibration import quafu_noise_model_from_chip_info
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

__all__ = (
    "AmazonBraketProvider",
    "AzureQuantumProvider",
    "AzureSubmissionPreview",
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
    "azure_backend_profile",
    "braket_backend_profile",
    "build_result_metadata",
    "build_submission_receipt",
    "quafu_noise_model_from_chip_info",
    "validate_deployment_result",
)
