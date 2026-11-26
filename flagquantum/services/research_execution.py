"""Historical import facade for the relocated implementation."""

from importlib import import_module as _import_module

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
    execute_research_capability as execute_research_capability,
)
from flagquantum.algorithms.optimization._research_execution import (
    validate_research_evidence as validate_research_evidence,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._research_execution"
)
