"""Independent MPS facade over its authoritative implementation."""

from __future__ import annotations

from .mps_execution import run_jax_sharded_mps
from .mps_gradient_result import JAXShardedMPSParameterGradientResult
from .mps_gradients import jax_sharded_mps_parameter_value_and_grad
from .mps_planning import plan_jax_sharded_mps_training
from .mps_result import JAXShardedMPSResult
from .mps_training_records import JAXMPSRankShardState, JAXShardedMPSTrainingPlan

__all__ = (
    "JAXMPSRankShardState",
    "JAXShardedMPSResult",
    "JAXShardedMPSParameterGradientResult",
    "JAXShardedMPSTrainingPlan",
    "run_jax_sharded_mps",
    "jax_sharded_mps_parameter_value_and_grad",
    "plan_jax_sharded_mps_training",
)
