"""Historical import facade for the relocated implementation."""

from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._research_capabilities import (
    EnergyVarianceRequest as EnergyVarianceRequest,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    ExactGroundStateRequest as ExactGroundStateRequest,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    ResearchCapabilityCandidate as ResearchCapabilityCandidate,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    ResearchCapabilityManifest as ResearchCapabilityManifest,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    ResearchCapabilityQuery as ResearchCapabilityQuery,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    ResearchResourceEstimate as ResearchResourceEstimate,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    ResearchValidationIssue as ResearchValidationIssue,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    ResearchValidationReport as ResearchValidationReport,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    __all__ as __all__,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    describe_research_capability as describe_research_capability,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    estimate_research_capability as estimate_research_capability,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    search_research_capabilities as search_research_capabilities,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    validate_research_capability_input as validate_research_capability_input,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._research_capabilities"
)
