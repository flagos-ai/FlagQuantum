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
    ProviderSubmissionPreview,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
    preflight_submission,
    validate_deployment_result,
)
from .http import (
    HttpQuantumProvider,
    ProviderCredentials,
    ProviderEndpoints,
    QuantumCloudTransport,
    UrllibTransport,
)
from .ionq import IONQ_GATE_SETS, IonQProvider, ionq_backend_profile
from .iqm import IQMProvider, iqm_backend_profile
from .neutral_atom import (
    NeutralAtomProvider,
    interaction_edges,
    neutral_atom_backend_profile,
)
from .quafu import QuafuProvider
from .quantinuum import QuantinuumProvider, quantinuum_backend_profile

__all__ = (
    "AmazonBraketProvider",
    "AzureQuantumProvider",
    "AzureSubmissionPreview",
    "BraketSubmissionPreview",
    "DEPLOYMENT_SUBMISSION_RECEIPT_SCHEMA",
    "DeploymentResult",
    "HttpQuantumProvider",
    "IONQ_GATE_SETS",
    "IQMProvider",
    "IonQProvider",
    "NeutralAtomProvider",
    "ProviderCredentials",
    "ProviderEndpoints",
    "ProviderSubmissionPreview",
    "ProviderTaskHandle",
    "QuafuProvider",
    "QuantinuumProvider",
    "QuantumCloudTransport",
    "QuantumProvider",
    "UrllibTransport",
    "azure_backend_profile",
    "braket_backend_profile",
    "build_result_metadata",
    "build_submission_receipt",
    "interaction_edges",
    "ionq_backend_profile",
    "iqm_backend_profile",
    "neutral_atom_backend_profile",
    "preflight_submission",
    "quafu_noise_model_from_chip_info",
    "quantinuum_backend_profile",
    "validate_deployment_result",
)
