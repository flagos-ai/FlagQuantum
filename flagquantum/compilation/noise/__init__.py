"""Backend-neutral compilation of noise models into channel IR."""

from .lowering import channel_instruction, lower_noise_model
from .planning import (
    EvolutionSemantics,
    MemoryPlan,
    NoiseErrorBudget,
    NoisyExecutionPlan,
    ParallelPlan,
    StateRepresentation,
    TrajectoryPlan,
    build_noisy_execution_plan,
)

__all__ = (
    "EvolutionSemantics",
    "MemoryPlan",
    "NoiseErrorBudget",
    "NoisyExecutionPlan",
    "ParallelPlan",
    "StateRepresentation",
    "TrajectoryPlan",
    "build_noisy_execution_plan",
    "channel_instruction",
    "lower_noise_model",
)
