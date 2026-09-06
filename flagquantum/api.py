# flagquantum/__init__.py
"""FlagQuantum - A distributed quantum computing framework.

FlagQuantum is a PyTorch-based quantum computing framework with built-in
support for distributed simulation across multiple GPUs.

Examples
--------
>>> import flagquantum as fq
>>> device = fq.DistributedQuantumDevice(n_wires=4, bsz=2)
>>> fq.h(device, wires=[0])
>>> fq.rx(device, wires=[1], params=0.5)
>>> fq.cx(device, wires=[0, 1])
>>> results = fq.measure_allZ(device)
"""

import logging

# Also expose submodules for advanced users
from . import (
    algorithms,
    compiler,
    deployment,
    devices,
    drawer,
    encoding,
    measurement,
    noise,
    ops,
    utils,
)
from ._compat_api_exports import compatibility_exports as _compatibility_exports
from .agent import (
    AgentExecutionPlan,
    DeploymentPreflightReport,
    ValidationIssue,
    ValidationReport,
    capabilities,
    preflight_deployment,
    preflight_execution,
    validate,
)
from .algorithms import (
    AdaptVQEIteration,
    AdaptVQEResult,
    Hamiltonian,
    HamiltonianTerm,
    VQEResult,
    hardware_efficient_ansatz,
    hardware_efficient_parameter_count,
    pauli_term,
    qaoa_circuit,
    qaoa_loss,
    run_adapt_vqe,
    run_vqe,
    transverse_field_ising,
    vqe_loss,
    zz_chain_hamiltonian,
)
from .circuit import Circuit, expectation
from .compiler import (
    CouplingMap,
    channel_instruction,
    lower_noise_model,
    route_to_topology,
)
from .core.ir import (
    IR_VERSION,
    CircuitIR,
    Instruction,
    IRSerializationError,
    IRValidationError,
    MeasurementNode,
    ObservableNode,
)
from .core.operator_schema import GateInfo, gate_info
from .core.parameters import Parameter, ParameterExpression
from .core.runtime_config import (
    RUNTIME_CONFIG_VERSION,
    RuntimeConfig,
    get_runtime_config,
    runtime_config,
    set_runtime_config,
)
from .deployment import (
    AmazonBraketProvider,
    BraketSubmissionPreview,
    CloudBackendProfile,
    DeploymentPackage,
    DeploymentResult,
    FieldQuantumProvider,
    GuodunProvider,
    HttpQuantumProvider,
    LocalSimulatorProvider,
    OriginQProvider,
    PauliMeasurementPlan,
    ProviderCredentials,
    ProviderEndpoints,
    ProviderTaskHandle,
    QuafuProvider,
    QuantumCloudTransport,
    QuantumProvider,
    TencentQuantumProvider,
    TianyanProvider,
    UrllibTransport,
    braket_backend_profile,
    create_deployment_package,
    create_pauli_measurement_plan,
    deploy_circuit,
    expectation_z_from_counts,
    hamiltonian_expectation_from_counts,
    hamiltonian_expectation_from_grouped_counts,
    quafu_noise_model_from_chip_info,
)
from .gradients import parameter_shift_gradient
from .models import HybridQuantumClassifier, VariationalEnergyModel
from .noise import (
    CorrelatedReadoutError,
    DeviceNoiseProfile,
    GateDuration,
    KrausChannel,
    NoiseModel,
    NoiseRule,
    QubitNoiseCalibration,
    ReadoutError,
    ReadoutRule,
    amplitude_damping_channel,
    bit_flip_channel,
    coherent_overrotation_channel,
    depolarizing_channel,
    phase_damping_channel,
    phase_flip_channel,
    reset_error_channel,
    thermal_relaxation_channel,
    two_qubit_depolarizing_channel,
)
from .runtime import planner
from .runtime.backends.statevector import (
    BatchedStatevectorTrajectoryResult,
    merge_noisy_statevector_results,
    run_noisy_statevector,
)
from .runtime.compatibility import (
    FLAGGEMS_EXPERIMENTAL_OPS,
    FLAGGEMS_FEATURE_REQUESTS,
    FLAGGEMS_NATIVE_PYTORCH_HOTSPOTS,
    FLAGGEMS_NON_OPERATOR_REQUIREMENTS,
    FLAGGEMS_SAFE_OPS,
    AcceleratorInfo,
    BackendCapabilities,
    DistributedBackendParityReport,
    DistributedBackendPolicy,
    DistributedBoundaryProtocol,
    DistributedBoundarySync,
    DistributedEvidenceContract,
    DistributedExecutor,
    DistributedIdentity,
    DistributedMPSState,
    DistributedScalabilityAudit,
    DistributedScalabilityError,
    DistributedShardPlan,
    DistributedSliceTask,
    DistributedStatevectorPlan,
    DistributedTensorNetworkState,
    DistributedTransportEvidence,
    ExecutionResult,
    HybridParallelPlan,
    JAXDistributedQuantumPlan,
    JAXMPSRankShardState,
    JAXQuantumKernel,
    JAXShardedMPSParameterFlowPlan,
    JAXShardedMPSParameterGateAssignment,
    JAXShardedMPSParameterGradientResult,
    JAXShardedMPSResult,
    JAXShardedMPSTrainingPlan,
    JAXShardedStatevectorParameterGradientResult,
    JAXShardedStatevectorResult,
    JAXShardedStatevectorTrainingPlan,
    JAXShardedTensorNetworkResult,
    JAXSlicedTensorNetworkGradientResult,
    JAXSlicedTensorNetworkParameterGradientResult,
    JAXStatevectorShardState,
    JAXTensorNetworkNode,
    JAXTNSliceRankState,
    LocalDistributedPreflightReport,
    LocalDistributedStatevectorResult,
    LocalFastPathPreflightReport,
    LocalTensor,
    LocalTensorShard,
    Module,
    MPSAcceptanceGates,
    MPSCrossoverMeasurement,
    MPSFullMaterializationError,
    MPSProductionAcceptanceError,
    MPSProductionPlan,
    MPSProductionSupport,
    MPSReverseCheckpointPolicy,
    MPSTrainingStep,
    NonFiniteTrainingError,
    NonlocalMPSCompilationError,
    ObservableGroup,
    OperatorBackendAvailability,
    OperatorBackendSession,
    OperatorReplacementPlan,
    Phase4StatevectorClaimabilityGate,
    PrecisionPolicy,
    PrecisionPolicyError,
    QuantumTorchLayer,
    RuntimePolicy,
    SeedContract,
    ShardedMPSState,
    StatevectorBufferPlan,
    StatevectorCommunicationEdge,
    StatevectorCorrectnessRunSpec,
    StatevectorExecutionSegment,
    StatevectorExecutorReport,
    StatevectorFusionBlock,
    StatevectorGatePlan,
    StatevectorPerformanceEstimate,
    StatevectorRankResult,
    StatevectorRankTopology,
    StatevectorSegmentResult,
    StatevectorShard,
    StatevectorShardState,
    StatevectorTraceEvent,
    StatevectorTraceReport,
    StatevectorTrainingClaimabilityGate,
    StatevectorTransportEvent,
    StatevectorTransportReport,
    TopologyMismatchError,
    TorchDistributedContext,
    TrainingResult,
    TrainingStateError,
    apply_gate_to_statevector_shard,
    apply_gate_to_statevector_shards,
    assert_finite_training,
    attach_distributed_evidence_contract,
    attach_distributed_scalability_audit,
    attach_sharded_optimizer_step_evidence,
    audit,
    audit_distributed_scalability,
    backend_execution_options,
    backends,
    build_mps_release_artifact,
    build_statevector_correctness_run_spec,
    capability_summary,
    compile_mps_training_step,
    compile_quantum_kernel,
    destroy_torch_distributed,
    detect_accelerators,
    distributed,
    distributed_backend_env_help,
    distributed_tensor_network_amplitude,
    distributed_tensor_network_amplitudes,
    distributed_tensor_network_expectation,
    distributed_tensor_network_expectations,
    estimate_distributed_statevector_performance,
    evaluate_distributed_evidence_contract,
    evaluate_distributed_transport_evidence,
    evaluate_phase4_statevector_claimability,
    evaluate_statevector_training_claimability,
    execute_distributed_statevector_dry_run,
    execute_distributed_statevector_transport,
    execute_torch_distributed_mps_forward,
    execution,
    flaggems_availability,
    flaggems_preflight,
    get_backend,
    get_backend_capabilities,
    get_dtype,
    group_observables,
    init_torch_distributed,
    initialize_jax_distributed,
    initialize_statevector_shard,
    jax_sharded_mps_parameter_value_and_grad,
    jax_sharded_statevector_parameter_value_and_grad,
    jax_sliced_tensor_network_parameter_value_and_grad,
    jax_sliced_tensor_network_value_and_grad,
    list_backends,
    load_training_checkpoint,
    local_distributed_development_preflight,
    local_fast_path_preflight,
    local_preflight,
    operator_backend,
    operator_backend_from_env,
    operator_backends,
    parity,
    plan_distributed_statevector,
    plan_hybrid_parallel,
    plan_jax_distributed_quantum_backend,
    plan_jax_sharded_mps_parameter_flow,
    plan_jax_sharded_mps_training,
    plan_jax_sharded_statevector_training,
    plan_operator_replacements,
    plan_production_mps,
    refresh_backend_registry,
    register_backend,
    require_development_production_parity,
    require_distributed_scalability,
    require_verified_flagcx,
    resolve_device,
    resolve_distributed_backend_policy,
    resolve_dtype,
    run,
    run_distributed,
    run_distributed_mps,
    run_distributed_tensor_network,
    run_jax_sharded_mps,
    run_jax_sharded_statevector,
    run_jax_sharded_tensor_network,
    run_native,
    runtime,
    runtime_backend,
    runtime_dtype,
    save_training_checkpoint,
    seed_everything,
    set_backend,
    set_dtype,
    simulate_distributed_statevector_local,
    torch_distributed_is_available,
    trace_distributed_statevector_plan,
    train,
    train_distributed_mps,
    train_distributed_statevector,
    validate_development_production_parity,
    validate_distributed_claim_evidence,
    validate_distributed_statevector_plan,
    validate_flaggems_ops,
    validate_production_mps_workload,
)
from .runtime.noise_registry import noisy_density_matrix
from .runtime.planner import (
    NOISE_SELECTOR_CALIBRATION_SCHEMA,
    ExecutionPlan,
    NoiseBackendCandidate,
    NoiseExecutionSelection,
    NoiseSelectorCalibration,
    NoiseSelectorCalibrationRecord,
    RuntimeCandidate,
    RuntimeSelectionPlan,
    analyze,
    estimate_density_bytes,
    estimate_mps_bytes,
    estimate_state_bytes,
    estimate_tensor_network_bytes,
    load_noise_selector_calibration,
    plan,
    plan_for_backend,
    plan_noise_execution_selection,
    plan_runtime_selection,
    select_backend_by_cost,
    select_execution_mode,
)
from .runtime.result import MeasurementResult
from .runtime.target_execution import TargetExecutionResult, run_target
from .simulation import graph, linalg, mps, tensor
from .simulation.density_matrix import (
    apply_kraus_density,
    apply_unitary_density,
    density_matrix,
    density_matrix_from_ir,
    expand_operator,
    expectation_z_density,
)
from .simulation.linalg import expm, random_unitary
from .simulation.mps import (
    MPSAdaptiveBondPlan,
    MPSAdaptiveRunResult,
    MPSBondProfile,
    MPSConfig,
    MPSLocalRefinementPlan,
    MPSMonteCarloResult,
    MPSState,
    MPSTruncationRecord,
    merge_noisy_mps_results,
    run_mps,
    run_mps_adaptive,
    run_noisy_mps,
    run_noisy_mps_trajectory,
)
from .simulation.tensor import (
    ContractionPathStep,
    PairContractionStep,
    TensorNetworkContractionPlan,
    TensorNetworkContractionProfile,
    TensorNetworkExpectationPlan,
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
    TensorNetworkState,
    build_tensor_network,
    build_tensor_network_expectation,
    build_tensor_network_hamiltonian_expectation,
    build_tensor_network_hamiltonian_expectations,
    run_tensor_network,
    tensor_network_amplitude,
    tensor_network_amplitudes,
    tensor_network_expectation_ps,
    tensor_network_expectations,
)
from .utils.qcis_exporter import QCISInstruction, export_to_qcis_str
from .version import __version__, get_version

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# ============================================================================
# Package info
# ============================================================================

__author__ = "FlagQuantum Team"
__license__ = "Apache-2.0"
# Historical module namespace retained lazily through the compatibility API.
invertible = ops.invertible

_COMPAT_EXPORT_MODULES = (devices, drawer, encoding, measurement, ops, utils)


def __getattr__(name):
    for module in _COMPAT_EXPORT_MODULES:
        if name in getattr(module, "__all__", ()):
            return getattr(module, name)
    raise AttributeError(name)


def info() -> dict:
    """Get package information."""
    return {
        "name": "flagquantum",
        "version": __version__,
        "author": __author__,
        "license": __license__,
    }


def hello() -> None:
    """Print welcome message."""
    print(f"FlagQuantum v{__version__} - Distributed Quantum Computing Framework")


# ============================================================================
# Module exports
# ============================================================================

__all__ = _compatibility_exports(
    devices=devices,
    drawer=drawer,
    encoding=encoding,
    measurement=measurement,
    ops=ops,
    utils=utils,
)
