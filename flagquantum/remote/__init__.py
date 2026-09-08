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
    "HttpQuantumProvider",
    "ProviderCredentials",
    "ProviderEndpoints",
    "QuafuProvider",
    "QuantumCloudTransport",
    "UrllibTransport",
    "braket_backend_profile",
    "quafu_noise_model_from_chip_info",
)
