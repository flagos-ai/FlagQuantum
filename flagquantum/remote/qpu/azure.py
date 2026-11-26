"""Historical import facade for the relocated implementation."""

from flagquantum.compute.remote.qpu.azure import (
    AzureQuantumProvider as AzureQuantumProvider,
)
from flagquantum.compute.remote.qpu.azure import (
    AzureSubmissionPreview as AzureSubmissionPreview,
)
from flagquantum.compute.remote.qpu.azure import __all__ as __all__
from flagquantum.compute.remote.qpu.azure import (
    azure_backend_profile as azure_backend_profile,
)
