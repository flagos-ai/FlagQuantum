"""Authoritative Runtime planning and backend-selection entry points."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ...compiler import compile as compile_program
from ...compiler import lower_noise_model as lower_noise_model
from ...compiler import schedule_layers as schedule_layers
from ...core.ir import CircuitIR, MeasurementNode
from ...errors import CapabilityError, ValidationError
from ..execution_plan import (
    CircuitAnalysis,
    EvolutionSemantics,
    ExecutionPlan,
    LayerPlan,
    MemoryPlan,
    NoiseErrorBudget,
    NoisyExecutionPlan,
    ParallelPlan,
    StateRepresentation,
    TrajectoryPlan,
)
from ..execution_plan_contract import build_layer_plans
from .backend_selection import (
    BackendCost,
    BackendSelection,
    OutputTarget,
    select_backend_by_cost,
)
from .candidates import (
    CandidateBuildContext,
    RuntimeCandidate,
    RuntimeCandidateBuilder,
)
from .estimates import (
    estimate_density_bytes,
    estimate_mps_bytes,
    estimate_state_bytes,
    estimate_tensor_network_bytes,
)
from .execution_policy import (
    estimate_execution_state_bytes,
    normalize_execution_state_mode,
    recommend_execution_mode,
)
from .metadata_projection import (
    _jax_mps_runtime_metadata_from_training_summary,
    _jax_statevector_runtime_metadata_from_training_summary,
)
from .models import RuntimeSelectionPlan
from .noise_calibration import (
    NOISE_SELECTOR_CALIBRATION_SCHEMA,
    NoiseSelectorCalibration,
    NoiseSelectorCalibrationRecord,
    load_noise_selector_calibration,
)
from .noise_selection import (
    NoiseBackendCandidate,
    NoiseExecutionSelection,
    plan_noise_execution_selection,
)
from .providers import (
    add_distributed_candidates,
    add_local_state_candidates,
    add_mps_candidates,
    add_tensor_network_candidates,
)
from .selection_context import build_runtime_selection_context
from .selection_result import finalize_runtime_selection
from .tn_calibration import (
    TNWorkingSetCalibration,
    build_tn_working_set_calibration,
    load_tn_working_set_calibration,
)
from .training_preflight import collect_distributed_training_preflight

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...noise import NoiseModel
    from ..options import ExecutionOptions


def analyze(ir: CircuitIR) -> CircuitAnalysis:
    """Analyze circuit structure without executing it."""

    gate_counts: dict[str, int] = {}
    wire_usage = [0] * ir.n_wires
    max_gate_width = 0
    two_qubit_gates = 0
    multi_qubit_gates = 0
    channel_count = 0

    for instruction in ir:
        gate_counts[instruction.name] = gate_counts.get(instruction.name, 0) + 1
        if instruction.metadata.get("is_channel"):
            channel_count += 1
        width = len(instruction.wires)
        max_gate_width = max(max_gate_width, width)
        if width == 2:
            two_qubit_gates += 1
        elif width > 2:
            multi_qubit_gates += 1
        for wire in instruction.wires:
            wire_usage[wire] += 1

    return CircuitAnalysis(
        n_wires=ir.n_wires,
        n_instructions=len(ir),
        depth=len(schedule_layers(ir)),
        gate_counts=gate_counts,
        wire_usage=tuple(wire_usage),
        max_gate_width=max_gate_width,
        two_qubit_gates=two_qubit_gates,
        multi_qubit_gates=multi_qubit_gates,
        channel_count=channel_count,
        has_noise=channel_count > 0,
    )


def build_execution_plan(
    *,
    ir: CircuitIR,
    analysis: CircuitAnalysis,
    state_bytes: int,
    recommended_mode: str,
    world_size: int,
    state_mode: str,
    runtime_config: Mapping[str, Any],
) -> ExecutionPlan:
    """Assemble the stable plan product from resolved Runtime policy."""

    distributed = world_size > 1
    routing_plan = dict(ir.metadata.get("routing", {}) or {})
    strategy_selection = ir.metadata.get("routing_strategy_selection")
    if strategy_selection:
        routing_plan["strategy_selection"] = strategy_selection
    return ExecutionPlan(
        analysis=analysis,
        layers=build_layer_plans(ir),
        state_bytes=state_bytes,
        recommended_mode=recommended_mode,
        world_size=int(world_size),
        shardable_wires=tuple(range(max(0, ir.n_wires - 1))),
        state_mode=state_mode,
        user_tier=("production_distributed" if distributed else "single_device"),
        usability_contract=(
            "single_api_distributed_scale_out"
            if distributed
            else "single_api_fast_path"
        ),
        runtime_config=runtime_config,
        routing_plan=routing_plan,
    )


def build_noisy_execution_plan(
    execution_plan: ExecutionPlan,
    *,
    representation: StateRepresentation,
    evolution: EvolutionSemantics,
    trajectories: int | None = None,
    seed: int | None = None,
    min_trajectories: int = 1,
    target_standard_error: float | None = None,
    cutoff: float = 0.0,
    memory_limit_bytes: int | None = None,
    estimated_memory_bytes: int | None = None,
    noise_model_identity: str | None = None,
) -> NoisyExecutionPlan:
    """Project an execution plan into the noisy Runtime plan product."""

    trajectory = (
        TrajectoryPlan(
            count=trajectories,
            seed=seed,
            min_count=min_trajectories,
            target_standard_error=target_standard_error,
        )
        if evolution == "quantum_trajectory" and trajectories is not None
        else None
    )
    return NoisyExecutionPlan(
        representation=representation,
        evolution=evolution,
        trajectory=trajectory,
        parallel=ParallelPlan(world_size=execution_plan.world_size),
        error_budget=NoiseErrorBudget(
            sampling_error_enabled=evolution == "quantum_trajectory",
            truncation_cutoff=cutoff,
        ),
        memory=MemoryPlan(
            estimated_bytes=(
                execution_plan.state_bytes
                if estimated_memory_bytes is None
                else int(estimated_memory_bytes)
            ),
            limit_bytes=memory_limit_bytes,
        ),
        noise_model_identity=noise_model_identity,
    )


def plan_runtime_selection(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    world_size: int = 1,
    local_world_size: int | None = None,
    node_count: int | None = None,
    complex_bytes: int = 8,
    memory_limit_bytes: int | None = None,
    noise_model: Any | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    prefer_jax: bool = False,
    prefer_distributed: bool | None = None,
    require_gradients: bool = True,
    require_deployment: bool = False,
    allow_approximate: bool = True,
    state_mode: str = "auto",
    max_intermediate_size: int | None = None,
    distributed_profile: str | None = None,
    jax_backward_backend: str = "auto",
    inspect_jax_devices: bool = False,
    assume_jax_devices_ready: bool = False,
    target: OutputTarget = "full_state",
    target_count: int = 1,
    tn_memory_calibration: TNWorkingSetCalibration | None = None,
    require_tn_memory_calibration: bool = False,
    accelerator_name: str | None = None,
) -> RuntimeSelectionPlan:
    """Explain how FlagQuantum should run one workload.

    This is an advisory product planner: it does not execute kernels, import JAX,
    initialize process groups, or turn rank-local acceleration into a scalability
    claim.  It gives UI, benchmarks, cloud deployment, and training code one
    shared place to ask "what should run, what is fast, and what is still
    blocked?"
    """

    ir = circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    analysis = analyze(ir)
    cost_selection = select_backend_by_cost(
        ir,
        target=target,
        require_gradients=require_gradients,
        world_size=world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        memory_limit_bytes=memory_limit_bytes,
        max_bond=max_bond,
        allow_approximate=allow_approximate,
        requested_backend=state_mode,
        target_count=target_count,
        tn_memory_calibration=tn_memory_calibration,
        require_tn_memory_calibration=require_tn_memory_calibration,
        accelerator_name=accelerator_name,
    )
    tn_cost = next(
        item for item in cost_selection.candidates if item.backend == "tensor_network"
    )
    selection = build_runtime_selection_context(
        analysis,
        bsz=bsz,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=node_count,
        complex_bytes=complex_bytes,
        max_bond=max_bond,
        max_intermediate_size=(
            max_intermediate_size
            if max_intermediate_size is not None
            else (tn_cost.estimated_memory_bytes if target != "full_state" else None)
        ),
        state_mode=state_mode,
        prefer_jax=prefer_jax,
        prefer_distributed=prefer_distributed,
        require_gradients=require_gradients,
        require_deployment=require_deployment,
    )
    world_size = selection.world_size
    local_world_size = selection.local_world_size
    node_count = selection.node_count
    prefer_distributed = selection.prefer_distributed
    dense_bytes = selection.dense_bytes
    density_bytes = selection.density_bytes
    mps_bytes = selection.mps_bytes
    tn_peak = selection.tensor_network_peak_bytes

    candidates: list[RuntimeCandidate] = []

    builder = RuntimeCandidateBuilder(
        CandidateBuildContext(
            analysis=analysis,
            candidates=candidates,
            require_gradients=require_gradients,
            require_deployment=require_deployment,
            memory_limit_bytes=memory_limit_bytes,
            density_bytes=density_bytes,
            mps_bytes=mps_bytes,
            tensor_network_peak_bytes=tn_peak,
            dense_bytes=dense_bytes,
            world_size=world_size,
            local_world_size=local_world_size,
            node_count=node_count,
            complex_bytes=complex_bytes,
            prefer_jax=prefer_jax,
            prefer_distributed=prefer_distributed,
            requested_state_mode=selection.requested_state_mode,
            statevector_metadata_projector=_jax_statevector_runtime_metadata_from_training_summary,
            mps_metadata_projector=_jax_mps_runtime_metadata_from_training_summary,
        )
    )
    noisy_warning = add_local_state_candidates(
        builder,
        analysis=analysis,
        world_size=world_size,
        dense_bytes=dense_bytes,
        density_bytes=density_bytes,
        require_gradients=require_gradients,
        noise_model_present=noise_model is not None,
    )
    mps_warnings = add_mps_candidates(
        builder,
        analysis=analysis,
        world_size=world_size,
        mps_bytes=mps_bytes,
        require_gradients=require_gradients,
        max_bond=max_bond,
        cutoff=cutoff,
        allow_approximate=allow_approximate,
        memory_limit_bytes=memory_limit_bytes,
        prefer_jax=prefer_jax,
        noisy_warning=noisy_warning,
    )
    add_tensor_network_candidates(
        builder,
        world_size=world_size,
        peak_bytes=tn_peak,
        require_gradients=require_gradients,
        prefer_jax=prefer_jax,
        noisy_warning=noisy_warning,
    )

    if world_size > 1 or prefer_distributed:
        training_preflight = collect_distributed_training_preflight(
            ir,
            enabled=require_gradients,
            world_size=world_size,
            local_world_size=local_world_size,
            bsz=bsz,
            complex_bytes=complex_bytes,
            max_bond=max_bond,
            cutoff=cutoff,
            backward_backend=jax_backward_backend,
            distributed_profile=distributed_profile,
            inspect_devices=inspect_jax_devices,
            assume_devices_ready=assume_jax_devices_ready,
        )
        add_distributed_candidates(
            builder,
            world_size=world_size,
            sharded_dense_bytes=selection.sharded_dense_bytes,
            mps_bytes=mps_bytes,
            tensor_network_peak_bytes=tn_peak,
            require_gradients=require_gradients,
            prefer_jax=prefer_jax,
            mps_warnings=mps_warnings,
            statevector_summary=training_preflight.statevector_summary,
            statevector_blockers=training_preflight.statevector_blockers,
            statevector_claim_allowed=training_preflight.statevector_claim_allowed,
            mps_summary=training_preflight.mps_summary,
            mps_blockers=training_preflight.mps_blockers,
        )

    cost_summary = cost_selection.summary()
    candidates = [
        replace(
            candidate,
            score=(
                candidate.score + 120
                if candidate.state_mode == cost_selection.selected_backend
                else candidate.score
            ),
            reasons=(
                (*candidate.reasons, "selected_by_output_and_structure_cost_model")
                if candidate.state_mode == cost_selection.selected_backend
                else candidate.reasons
            ),
            warnings=(
                (*candidate.warnings, *cost_selection.warnings)
                if candidate.state_mode == cost_selection.selected_backend
                else candidate.warnings
            ),
            metadata={
                **dict(candidate.metadata or {}),
                "backend_cost_selection": cost_summary,
            },
        )
        for candidate in candidates
    ]

    return finalize_runtime_selection(
        analysis=analysis,
        candidates=candidates,
        objective=selection.objective,
        prefer_distributed=prefer_distributed,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=node_count,
    )


def select_execution_mode(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    world_size: int = 1,
    complex_bytes: int = 8,
    memory_limit_bytes: int | None = None,
    noise_model: Any | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    trajectories: int | None = None,
    trajectory_batch_size: int = 32,
    target: OutputTarget = "full_state",
    require_gradients: bool = False,
    allow_approximate: bool = True,
    target_count: int = 1,
    noise_performance_calibration: Any | None = None,
    min_trajectories: int = 1,
    target_standard_error: float | None = None,
    pilot_variance: float | None = None,
    pilot_trajectories: int | None = None,
    pilot_confidence_level: float | None = None,
    pilot_observable_count: int = 1,
) -> str:
    """Choose the native execution mode for automatic dispatch."""

    ir = circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    if noise_model is not None:
        return plan_noise_execution_selection(
            ir,
            noise_model,
            bsz=bsz,
            world_size=world_size,
            complex_bytes=complex_bytes,
            memory_limit_bytes=memory_limit_bytes,
            trajectories=trajectories,
            trajectory_batch_size=trajectory_batch_size,
            max_bond=max_bond,
            cutoff=cutoff,
            allow_approximate=allow_approximate,
            calibration=noise_performance_calibration,
            min_trajectories=min_trajectories,
            target_standard_error=target_standard_error,
            pilot_variance=pilot_variance,
            pilot_trajectories=pilot_trajectories,
            pilot_confidence_level=pilot_confidence_level,
            pilot_observable_count=pilot_observable_count,
        ).selected_mode
    selection = select_backend_by_cost(
        ir,
        target=target,
        require_gradients=require_gradients,
        world_size=world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        memory_limit_bytes=memory_limit_bytes,
        max_bond=max_bond,
        allow_approximate=allow_approximate,
        target_count=target_count,
    )
    selected = selection.selected_backend
    if int(world_size) > 1:
        if selected == "statevector":
            return "distributed_statevector"
        if selected == "mps":
            return "distributed_mps"
        if selected == "tensor_network":
            return "distributed_tensor_network"
    return selected


def plan_advanced(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    world_size: int = 1,
    complex_bytes: int = 8,
    memory_limit_bytes: int | None = None,
    state_mode: str = "auto",
    noise_model: Any | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    trajectories: int | None = None,
    target: OutputTarget = "full_state",
    require_gradients: bool = False,
    allow_approximate: bool = True,
    target_count: int = 1,
    coupling_map: Any | None = None,
    routing_strategy: str = "restore_after_each_gate",
    optimize: bool = True,
    config: Any | None = None,
) -> ExecutionPlan:
    """Build an expert execution plan with backend-specific controls.

    This function is intentionally outside the Stable Core. Normal callers
    should use :func:`plan` with an ``ExecutionOptions`` object.
    """

    ir = circuit_or_ir.to_ir() if hasattr(circuit_or_ir, "to_ir") else circuit_or_ir
    from ...core.runtime_config import get_runtime_config

    selected_config = config or get_runtime_config()
    ir = compile_program(
        ir,
        coupling_map=coupling_map,
        routing_strategy=routing_strategy,
        optimize=optimize,
        config=selected_config,
    )
    if noise_model is not None:
        ir = lower_noise_model(ir, noise_model)
    analysis = analyze(ir)
    auto_selected_mode = None
    if state_mode == "auto":
        auto_selected_mode = select_execution_mode(
            circuit_or_ir,
            bsz=bsz,
            world_size=world_size,
            complex_bytes=complex_bytes,
            memory_limit_bytes=memory_limit_bytes,
            noise_model=noise_model,
            max_bond=max_bond,
            cutoff=cutoff,
            trajectories=trajectories,
            target=target,
            require_gradients=require_gradients,
            allow_approximate=allow_approximate,
            target_count=target_count,
        )
    state_mode = normalize_execution_state_mode(
        state_mode,
        auto_selected_mode=auto_selected_mode,
    )
    state_bytes = estimate_execution_state_bytes(
        state_mode,
        n_wires=ir.n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
        max_bond=max_bond,
    )
    recommended_mode = recommend_execution_mode(
        state_mode,
        world_size=world_size,
        has_noise=analysis.has_noise,
        state_bytes=state_bytes,
        memory_limit_bytes=memory_limit_bytes,
    )

    return build_execution_plan(
        ir=ir,
        analysis=analysis,
        state_bytes=state_bytes,
        recommended_mode=recommended_mode,
        world_size=world_size,
        state_mode=state_mode,
        runtime_config=selected_config.to_manifest(),
    )


def _validated_plan_program(
    program: Circuit | CircuitIR,
    *,
    measurements: Sequence[MeasurementNode] | None,
    noise_model: NoiseModel | None,
) -> CircuitIR:
    source_ir = program.to_ir() if hasattr(program, "to_ir") else program
    if not isinstance(source_ir, CircuitIR):
        raise TypeError("program must be a Circuit or CircuitIR")
    if measurements is not None:
        if source_ir.measurements:
            raise ValidationError(
                "measurements cannot be supplied when the program already contains "
                "measurement requests"
            )
        if any(not isinstance(request, MeasurementNode) for request in measurements):
            raise TypeError("measurements must contain MeasurementNode instances")
        source_ir = replace(source_ir, measurements=tuple(measurements))
    if noise_model is not None:
        from ...noise import NoiseModel

        if not isinstance(noise_model, NoiseModel):
            raise TypeError(
                "noise_model must be a flagquantum.noise.NoiseModel or None"
            )
    if any(
        instruction.metadata.get("is_dynamic") or instruction.metadata.get("conditions")
        for instruction in source_ir.instructions
    ):
        raise CapabilityError(
            "fq.run does not execute dynamic trajectories; use "
            "fq.experimental.dynamic.run_dynamic(..., shots=...)"
        )
    return source_ir


def plan(
    program: Circuit | CircuitIR,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
    noise_model: NoiseModel | None = None,
) -> ExecutionPlan:
    """Build an execution plan from the stable, backend-neutral options.

    Examples:
        Inspect the selected execution mode before running a circuit:

        >>> import flagquantum as fq
        >>> plan = fq.plan(fq.Circuit(2).h(0).cx(0, 1))
        >>> plan.state_mode
        'statevector'
    """

    from ..distributed.backend_policy import (
        resolve_distributed_backend_policy,
    )
    from ..options import ExecutionOptions
    from ..options_resolver import (
        circuit_execution_constraints,
        resolve_execution_options,
    )

    if options is not None and not isinstance(options, ExecutionOptions):
        raise TypeError("options must be an ExecutionOptions or None")
    source_ir = _validated_plan_program(
        program,
        measurements=measurements,
        noise_model=noise_model,
    )
    runtime_config = getattr(program, "runtime_config", None)
    resolved = resolve_execution_options(
        options,
        program_constraints=circuit_execution_constraints(program),
        runtime_config=runtime_config,
    )
    if resolved.backend not in {"auto", "pytorch"}:
        raise CapabilityError(
            f"stable fq.run backend {resolved.backend!r} is not available; "
            "use fq.Module for JAX kernels or the owning Runtime or Simulation "
            "expert interface for backend-native execution"
        )
    if noise_model is not None and resolved.mode not in {"auto", "density_matrix"}:
        raise ValidationError(
            "stable noisy execution supports mode='auto' or mode='density_matrix'"
        )
    from ...core.runtime_config import get_runtime_config

    selected_config = runtime_config or get_runtime_config()
    selected_config = selected_config.with_overrides(
        backend=resolved.backend,
        device=resolved.device,
        complex_dtype=resolved.precision,
    )
    targets: dict[str, OutputTarget] = {
        "auto": "full_state",
        "state": "full_state",
        "expectation": "expectation",
        "samples": "samples",
        "amplitudes": "few_amplitudes",
    }
    measurements = tuple(source_ir.measurements)
    if not measurements and (
        resolved.target == "samples" or resolved.shots is not None
    ):
        if resolved.shots is None:
            raise ValidationError("target='samples' requires shots")
        measurements = (
            MeasurementNode(
                "sample",
                tuple(range(source_ir.n_wires)),
                shots=resolved.shots,
                metadata={} if resolved.seed is None else {"seed": resolved.seed},
            ),
        )
        source_ir = replace(source_ir, measurements=measurements)
    if not measurements and resolved.target in {"expectation", "amplitudes"}:
        raise CapabilityError(
            f"target={resolved.target!r} requires a measurement embedded in the program"
        )
    world_size = resolve_distributed_backend_policy().effective_world_size
    internal_plan = plan_advanced(
        source_ir,
        bsz=resolved.batch_size,
        world_size=world_size,
        complex_bytes=16 if resolved.precision == "complex128" else 8,
        memory_limit_bytes=resolved.memory_limit_bytes,
        state_mode=resolved.mode,
        noise_model=noise_model,
        target=targets[resolved.target],
        require_gradients=resolved.require_gradients,
        allow_approximate=resolved.allow_approximate,
        config=selected_config,
    )
    if noise_model is not None:
        internal_plan = replace(
            internal_plan,
            noisy_execution_plan=build_noisy_execution_plan(
                internal_plan,
                representation="density_matrix",
                evolution="exact_channel",
                memory_limit_bytes=resolved.memory_limit_bytes,
                noise_model_identity=noise_model.identity,
            ),
        )
    from ..execution_plan_contract import attach_execution_contract

    return attach_execution_contract(
        internal_plan,
        program=source_ir,
        requested_options=options or ExecutionOptions(),
        resolved_options=resolved,
        noise_model=noise_model,
        preserve_program_instructions=noise_model is not None,
    )


def plan_for_backend(
    circuit_or_ir: Any,
    *,
    backend: str | None = None,
    device: str | None = "auto",
    dtype: str | None = None,
    mode: str = "auto",
    **options: Any,
) -> ExecutionPlan:
    """Build an execution plan after normalizing backend policy."""

    from ..planner_adapter import backend_execution_options

    backend_options = backend_execution_options(
        mode=mode,
        backend=backend,
        device=device,
        dtype=dtype,
        world_size=int(options.get("world_size", options.get("world_sz", 1))),
    )
    normalized_mode = "distributed_statevector" if mode == "distributed" else mode
    state_mode = (
        normalized_mode
        if normalized_mode
        in {
            "statevector",
            "density_matrix",
            "mps",
            "adaptive_mps",
            "distributed_mps",
            "tensor_network",
            "distributed_tensor_network",
            "tn",
        }
        else "auto"
    )
    complex_bytes = (
        16 if str(backend_options["complex_dtype"]).endswith("complex128") else 8
    )
    return plan_advanced(
        circuit_or_ir,
        state_mode=state_mode,
        complex_bytes=complex_bytes,
        world_size=backend_options["world_size"],
        **options,
    )


__all__ = [
    "BackendCost",
    "BackendSelection",
    "NOISE_SELECTOR_CALIBRATION_SCHEMA",
    "LayerPlan",
    "CircuitAnalysis",
    "ExecutionPlan",
    "NoiseBackendCandidate",
    "NoiseExecutionSelection",
    "NoiseSelectorCalibration",
    "NoiseSelectorCalibrationRecord",
    "OutputTarget",
    "RuntimeCandidate",
    "RuntimeSelectionPlan",
    "analyze",
    "build_execution_plan",
    "build_noisy_execution_plan",
    "build_tn_working_set_calibration",
    "estimate_density_bytes",
    "estimate_mps_bytes",
    "estimate_tensor_network_bytes",
    "estimate_state_bytes",
    "load_noise_selector_calibration",
    "load_tn_working_set_calibration",
    "plan",
    "plan_advanced",
    "plan_for_backend",
    "plan_noise_execution_selection",
    "plan_runtime_selection",
    "select_backend_by_cost",
    "select_execution_mode",
]
