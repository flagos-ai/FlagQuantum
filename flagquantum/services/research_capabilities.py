"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._research_capabilities import (
    _EXACT_ID as _EXACT_ID,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _MANIFESTS as _MANIFESTS,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _MANIFESTS_BY_ID as _MANIFESTS_BY_ID,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _MATURITY_BINDING as _MATURITY_BINDING,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _VARIANCE_ID as _VARIANCE_ID,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _VQE_ID as _VQE_ID,
)
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
    _closed_mapping as _closed_mapping,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _normalize_research_request as _normalize_research_request,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _normalized_query as _normalized_query,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _pauli_sum_from_mapping as _pauli_sum_from_mapping,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _statevector_from_mapping as _statevector_from_mapping,
)
from flagquantum.algorithms.optimization._research_capabilities import (
    _unsupported_report as _unsupported_report,
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
_sys.modules[__name__] = _implementation
