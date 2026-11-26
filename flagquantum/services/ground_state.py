"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.optimization.ground_state import (
    _ANSATZ_NAME as _ANSATZ_NAME,
)
from flagquantum.algorithms.optimization.ground_state import (
    _BASIS_CONVENTION as _BASIS_CONVENTION,
)
from flagquantum.algorithms.optimization.ground_state import (
    _OPTIMIZER_NAME as _OPTIMIZER_NAME,
)
from flagquantum.algorithms.optimization.ground_state import (
    _REFERENCE_SCHEMA as _REFERENCE_SCHEMA,
)
from flagquantum.algorithms.optimization.ground_state import (
    _REQUEST_SCHEMA as _REQUEST_SCHEMA,
)
from flagquantum.algorithms.optimization.ground_state import (
    _RESULT_SCHEMA as _RESULT_SCHEMA,
)
from flagquantum.algorithms.optimization.ground_state import (
    _TOOL_NAME as _TOOL_NAME,
)
from flagquantum.algorithms.optimization.ground_state import (
    _TOOL_VERSION as _TOOL_VERSION,
)
from flagquantum.algorithms.optimization.ground_state import (
    ComplexAmplitude as ComplexAmplitude,
)
from flagquantum.algorithms.optimization.ground_state import (
    ConvergencePoint as ConvergencePoint,
)
from flagquantum.algorithms.optimization.ground_state import (
    EnergyVarianceReference as EnergyVarianceReference,
)
from flagquantum.algorithms.optimization.ground_state import (
    ExactSectorReference as ExactSectorReference,
)
from flagquantum.algorithms.optimization.ground_state import (
    ExecutionProvenance as ExecutionProvenance,
)
from flagquantum.algorithms.optimization.ground_state import (
    GroundStateAnsatz as GroundStateAnsatz,
)
from flagquantum.algorithms.optimization.ground_state import (
    GroundStateOptimizer as GroundStateOptimizer,
)
from flagquantum.algorithms.optimization.ground_state import (
    GroundStateResources as GroundStateResources,
)
from flagquantum.algorithms.optimization.ground_state import (
    GroundStateVQERequest as GroundStateVQERequest,
)
from flagquantum.algorithms.optimization.ground_state import (
    GroundStateVQEResult as GroundStateVQEResult,
)
from flagquantum.algorithms.optimization.ground_state import (
    PauliExpectationEvidence as PauliExpectationEvidence,
)
from flagquantum.algorithms.optimization.ground_state import (
    PauliSum as PauliSum,
)
from flagquantum.algorithms.optimization.ground_state import (
    PauliSumTerm as PauliSumTerm,
)
from flagquantum.algorithms.optimization.ground_state import (
    StatevectorEvidence as StatevectorEvidence,
)
from flagquantum.algorithms.optimization.ground_state import (
    UnsupportedCapability as UnsupportedCapability,
)
from flagquantum.algorithms.optimization.ground_state import __all__ as __all__
from flagquantum.algorithms.optimization.ground_state import (
    _canonical_digest as _canonical_digest,
)
from flagquantum.algorithms.optimization.ground_state import (
    _native_hamiltonian as _native_hamiltonian,
)
from flagquantum.algorithms.optimization.ground_state import (
    _provenance as _provenance,
)
from flagquantum.algorithms.optimization.ground_state import (
    _unsupported_reason as _unsupported_reason,
)
from flagquantum.algorithms.optimization.ground_state import (
    _unsupported_vqe_result as _unsupported_vqe_result,
)
from flagquantum.algorithms.optimization.ground_state import (
    energy_variance_reference as energy_variance_reference,
)
from flagquantum.algorithms.optimization.ground_state import (
    exact_sector_reference as exact_sector_reference,
)
from flagquantum.algorithms.optimization.ground_state import (
    ground_state_vqe_capability as ground_state_vqe_capability,
)
from flagquantum.algorithms.optimization.ground_state import (
    run_ground_state_vqe as run_ground_state_vqe,
)
from flagquantum.algorithms.optimization.ground_state import (
    validate_ground_state_vqe_request as validate_ground_state_vqe_request,
)

_implementation = _import_module("flagquantum.algorithms.optimization.ground_state")
_sys.modules[__name__] = _implementation
