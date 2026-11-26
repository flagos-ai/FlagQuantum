"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.qpu.azure import (
    _BIT as _BIT,
)
from flagquantum.compute.remote.qpu.azure import (
    AzureQuantumProvider as AzureQuantumProvider,
)
from flagquantum.compute.remote.qpu.azure import (
    AzureSubmissionPreview as AzureSubmissionPreview,
)
from flagquantum.compute.remote.qpu.azure import __all__ as __all__
from flagquantum.compute.remote.qpu.azure import (
    _attribute as _attribute,
)
from flagquantum.compute.remote.qpu.azure import (
    _azure_counts as _azure_counts,
)
from flagquantum.compute.remote.qpu.azure import (
    _bitstring as _bitstring,
)
from flagquantum.compute.remote.qpu.azure import (
    _job_id as _job_id,
)
from flagquantum.compute.remote.qpu.azure import (
    _probabilities_to_counts as _probabilities_to_counts,
)
from flagquantum.compute.remote.qpu.azure import (
    azure_backend_profile as azure_backend_profile,
)

_implementation = _import_module("flagquantum.compute.remote.qpu.azure")
_sys.modules[__name__] = _implementation
