"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._research_execution import (
    _CONVERGENCE_TOLERANCE as _CONVERGENCE_TOLERANCE,
)
from flagquantum.algorithms.optimization._research_execution import (
    _CONVERGENCE_WINDOW as _CONVERGENCE_WINDOW,
)
from flagquantum.algorithms.optimization._research_execution import (
    _EXACT_ID as _EXACT_ID,
)
from flagquantum.algorithms.optimization._research_execution import (
    _VARIANCE_ID as _VARIANCE_ID,
)
from flagquantum.algorithms.optimization._research_execution import (
    _VQE_ID as _VQE_ID,
)
from flagquantum.algorithms.optimization._research_execution import (
    ResearchCapabilityAttempt as ResearchCapabilityAttempt,
)
from flagquantum.algorithms.optimization._research_execution import (
    ResearchCapabilityError as ResearchCapabilityError,
)
from flagquantum.algorithms.optimization._research_execution import (
    ResearchCapabilityEvidenceBundle as ResearchCapabilityEvidenceBundle,
)
from flagquantum.algorithms.optimization._research_execution import (
    ResearchCapabilityRef as ResearchCapabilityRef,
)
from flagquantum.algorithms.optimization._research_execution import (
    _canonical_sha256 as _canonical_sha256,
)
from flagquantum.algorithms.optimization._research_execution import (
    _capability_ref as _capability_ref,
)
from flagquantum.algorithms.optimization._research_execution import (
    _error_from_validation as _error_from_validation,
)
from flagquantum.algorithms.optimization._research_execution import (
    _evidence_sha256 as _evidence_sha256,
)
from flagquantum.algorithms.optimization._research_execution import (
    _execute_exact as _execute_exact,
)
from flagquantum.algorithms.optimization._research_execution import (
    _execute_variance as _execute_variance,
)
from flagquantum.algorithms.optimization._research_execution import (
    _execute_vqe as _execute_vqe,
)
from flagquantum.algorithms.optimization._research_execution import (
    _finalize_bundle as _finalize_bundle,
)
from flagquantum.algorithms.optimization._research_execution import (
    _installed_vqe_providers as _installed_vqe_providers,
)
from flagquantum.algorithms.optimization._research_execution import (
    _invalid_evidence as _invalid_evidence,
)
from flagquantum.algorithms.optimization._research_execution import (
    _now as _now,
)
from flagquantum.algorithms.optimization._research_execution import (
    _portable_evidence_value as _portable_evidence_value,
)
from flagquantum.algorithms.optimization._research_execution import (
    _provenance_base as _provenance_base,
)
from flagquantum.algorithms.optimization._research_execution import (
    _refused_bundle as _refused_bundle,
)
from flagquantum.algorithms.optimization._research_execution import (
    _request_sha256 as _request_sha256,
)
from flagquantum.algorithms.optimization._research_execution import (
    _route_vqe as _route_vqe,
)
from flagquantum.algorithms.optimization._research_execution import (
    _scientific_report as _scientific_report,
)
from flagquantum.algorithms.optimization._research_execution import (
    _unsigned_payload as _unsigned_payload,
)
from flagquantum.algorithms.optimization._research_execution import (
    execute_research_capability as execute_research_capability,
)
from flagquantum.algorithms.optimization._research_execution import (
    validate_research_evidence as validate_research_evidence,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._research_execution"
)
_sys.modules[__name__] = _implementation
