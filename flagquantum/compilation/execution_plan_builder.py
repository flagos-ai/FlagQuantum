"""Structural construction helpers for backend-neutral execution plans."""

from __future__ import annotations

from typing import Any, Mapping

from ..compiler import schedule_layers
from ..core.ir import CircuitIR
from .models import (
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


def build_layer_plans(ir: CircuitIR) -> tuple[LayerPlan, ...]:
    """Convert the scheduler output into immutable layer descriptions."""

    return tuple(
        LayerPlan(
            index=index,
            instructions=tuple(layer),
            wires=tuple(
                sorted({wire for instruction in layer for wire in instruction.wires})
            ),
        )
        for index, layer in enumerate(schedule_layers(ir))
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
    """Assemble the stable execution-plan model from computed policy."""

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
    """Project an execution plan into the noisy-runtime plan contract."""

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


__all__ = (
    "build_execution_plan",
    "build_layer_plans",
    "build_noisy_execution_plan",
)
