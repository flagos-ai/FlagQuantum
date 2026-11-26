"""Maintained public workflow imports; implementations live in their owning domains."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flagquantum._api import capabilities as capabilities
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
    from flagquantum.algorithms.optimization._research_options import (
        ResearchExecutionOptions as ResearchExecutionOptions,
    )
    from flagquantum.algorithms.optimization._research_schemas import (
        ResearchSchemaDocument as ResearchSchemaDocument,
    )
    from flagquantum.algorithms.optimization._research_schemas import (
        describe_research_schema as describe_research_schema,
    )
    from flagquantum.algorithms.optimization._research_schemas import (
        list_research_schemas as list_research_schemas,
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
    from flagquantum.algorithms.optimization.ground_state import PauliSum as PauliSum
    from flagquantum.algorithms.optimization.ground_state import (
        PauliSumTerm as PauliSumTerm,
    )
    from flagquantum.algorithms.optimization.ground_state import (
        StatevectorEvidence as StatevectorEvidence,
    )
    from flagquantum.algorithms.optimization.ground_state import (
        UnsupportedCapability as UnsupportedCapability,
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
    from flagquantum.runtime.managed_simulator import (
        ManagedSimulatorExecution as ManagedSimulatorExecution,
    )
    from flagquantum.runtime.managed_simulator import (
        ManagedSimulatorReceipt as ManagedSimulatorReceipt,
    )
    from flagquantum.runtime.managed_simulator import (
        managed_quafu_simulator_device as managed_quafu_simulator_device,
    )
    from flagquantum.runtime.managed_simulator import (
        run_managed_quafu_simulator as run_managed_quafu_simulator,
    )
    from flagquantum.runtime.planner.preflight import (
        DeploymentPreflightReport as DeploymentPreflightReport,
    )
    from flagquantum.runtime.planner.preflight import (
        ExecutionPreflightReport as ExecutionPreflightReport,
    )
    from flagquantum.runtime.planner.preflight import ValidationIssue as ValidationIssue
    from flagquantum.runtime.planner.preflight import (
        ValidationReport as ValidationReport,
    )
    from flagquantum.runtime.planner.preflight import (
        preflight_deployment as preflight_deployment,
    )
    from flagquantum.runtime.planner.preflight import (
        preflight_execution as preflight_execution,
    )

_OWNERS = {
    "ComplexAmplitude": "flagquantum.services.ground_state",
    "ConvergencePoint": "flagquantum.services.ground_state",
    "EnergyVarianceReference": "flagquantum.services.ground_state",
    "ExactSectorReference": "flagquantum.services.ground_state",
    "ExecutionProvenance": "flagquantum.services.ground_state",
    "GroundStateAnsatz": "flagquantum.services.ground_state",
    "GroundStateOptimizer": "flagquantum.services.ground_state",
    "GroundStateResources": "flagquantum.services.ground_state",
    "GroundStateVQERequest": "flagquantum.services.ground_state",
    "GroundStateVQEResult": "flagquantum.services.ground_state",
    "PauliExpectationEvidence": "flagquantum.services.ground_state",
    "PauliSum": "flagquantum.services.ground_state",
    "PauliSumTerm": "flagquantum.services.ground_state",
    "StatevectorEvidence": "flagquantum.services.ground_state",
    "UnsupportedCapability": "flagquantum.services.ground_state",
    "energy_variance_reference": "flagquantum.services.ground_state",
    "exact_sector_reference": "flagquantum.services.ground_state",
    "ground_state_vqe_capability": "flagquantum.services.ground_state",
    "run_ground_state_vqe": "flagquantum.services.ground_state",
    "validate_ground_state_vqe_request": "flagquantum.services.ground_state",
    "ManagedSimulatorExecution": "flagquantum.services.managed_simulator",
    "ManagedSimulatorReceipt": "flagquantum.services.managed_simulator",
    "managed_quafu_simulator_device": "flagquantum.services.managed_simulator",
    "run_managed_quafu_simulator": "flagquantum.services.managed_simulator",
    "DeploymentPreflightReport": "flagquantum.services.preflight",
    "ExecutionPreflightReport": "flagquantum.services.preflight",
    "ValidationIssue": "flagquantum.services.preflight",
    "ValidationReport": "flagquantum.services.preflight",
    "capabilities": "flagquantum.services.preflight",
    "preflight_deployment": "flagquantum.services.preflight",
    "preflight_execution": "flagquantum.services.preflight",
    "EnergyVarianceRequest": "flagquantum.services.research_capabilities",
    "ExactGroundStateRequest": "flagquantum.services.research_capabilities",
    "ResearchCapabilityCandidate": "flagquantum.services.research_capabilities",
    "ResearchCapabilityManifest": "flagquantum.services.research_capabilities",
    "ResearchCapabilityQuery": "flagquantum.services.research_capabilities",
    "ResearchResourceEstimate": "flagquantum.services.research_capabilities",
    "ResearchValidationIssue": "flagquantum.services.research_capabilities",
    "ResearchValidationReport": "flagquantum.services.research_capabilities",
    "describe_research_capability": "flagquantum.services.research_capabilities",
    "estimate_research_capability": "flagquantum.services.research_capabilities",
    "search_research_capabilities": "flagquantum.services.research_capabilities",
    "validate_research_capability_input": "flagquantum.services.research_capabilities",
    "ResearchCapabilityAttempt": "flagquantum.services.research_execution",
    "ResearchCapabilityError": "flagquantum.services.research_execution",
    "ResearchCapabilityEvidenceBundle": "flagquantum.services.research_execution",
    "ResearchCapabilityRef": "flagquantum.services.research_execution",
    "execute_research_capability": "flagquantum.services.research_execution",
    "validate_research_evidence": "flagquantum.services.research_execution",
    "ResearchExecutionOptions": "flagquantum.services.research_options",
    "ResearchSchemaDocument": "flagquantum.services.research_schemas",
    "describe_research_schema": "flagquantum.services.research_schemas",
    "list_research_schemas": "flagquantum.services.research_schemas",
}

__all__ = (
    "ComplexAmplitude",
    "ConvergencePoint",
    "EnergyVarianceReference",
    "EnergyVarianceRequest",
    "ExactSectorReference",
    "ExactGroundStateRequest",
    "ExecutionProvenance",
    "GroundStateAnsatz",
    "GroundStateOptimizer",
    "GroundStateResources",
    "GroundStateVQERequest",
    "GroundStateVQEResult",
    "ManagedSimulatorExecution",
    "ManagedSimulatorReceipt",
    "PauliExpectationEvidence",
    "PauliSum",
    "PauliSumTerm",
    "ResearchCapabilityCandidate",
    "ResearchCapabilityAttempt",
    "ResearchCapabilityError",
    "ResearchCapabilityEvidenceBundle",
    "ResearchCapabilityManifest",
    "ResearchCapabilityQuery",
    "ResearchCapabilityRef",
    "ResearchExecutionOptions",
    "ResearchResourceEstimate",
    "ResearchSchemaDocument",
    "ResearchValidationIssue",
    "ResearchValidationReport",
    "StatevectorEvidence",
    "UnsupportedCapability",
    "ExecutionPreflightReport",
    "DeploymentPreflightReport",
    "ValidationIssue",
    "ValidationReport",
    "capabilities",
    "describe_research_capability",
    "describe_research_schema",
    "energy_variance_reference",
    "estimate_research_capability",
    "execute_research_capability",
    "exact_sector_reference",
    "ground_state_vqe_capability",
    "managed_quafu_simulator_device",
    "list_research_schemas",
    "preflight_deployment",
    "preflight_execution",
    "run_managed_quafu_simulator",
    "run_ground_state_vqe",
    "search_research_capabilities",
    "validate_ground_state_vqe_request",
    "validate_research_capability_input",
    "validate_research_evidence",
)


def __getattr__(name: str) -> Any:
    if name in _OWNERS:
        value = getattr(import_module(_OWNERS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
