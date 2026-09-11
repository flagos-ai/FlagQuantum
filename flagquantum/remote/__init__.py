"""Compute targets reached through an external task control plane.

Remote adapters submit work and translate external task identifiers, status,
and results. Directly controlled CPU, GPU, and accelerator resources belong in
``flagquantum.compute``.
"""

from .qpu import (
    DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA,
    AmazonBraketProvider,
    BraketSubmissionPreview,
    DeploymentResult,
    HttpQuantumProvider,
    ProviderCredentials,
    ProviderEndpoints,
    ProviderTaskHandle,
    QuafuProvider,
    QuantumCloudTransport,
    QuantumProvider,
    UrllibTransport,
    braket_backend_profile,
    build_result_metadata,
    build_submission_receipt,
    quafu_noise_model_from_chip_info,
    validate_deployment_result,
)

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
