"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._research_providers import (
    NativeResearchVQEProvider as NativeResearchVQEProvider,
)
from flagquantum.algorithms.optimization._research_providers import (
    ResearchProviderAvailability as ResearchProviderAvailability,
)
from flagquantum.algorithms.optimization._research_providers import (
    ResearchVQEProvider as ResearchVQEProvider,
)
from flagquantum.algorithms.optimization._research_providers import __all__ as __all__
from flagquantum.algorithms.optimization._research_providers import (
    denied_vqe_provider_route as denied_vqe_provider_route,
)
from flagquantum.algorithms.optimization._research_providers import (
    installed_vqe_providers as installed_vqe_providers,
)
from flagquantum.algorithms.optimization._research_providers import (
    plan_vqe_provider_route as plan_vqe_provider_route,
)
from flagquantum.algorithms.optimization._research_providers import (
    research_vqe_metrics as research_vqe_metrics,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._research_providers"
)
_sys.modules[__name__] = _implementation
