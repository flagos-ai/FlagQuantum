"""Independent statevector facade over its authoritative implementation."""

from __future__ import annotations

from .statevector_execution import (
    jax_sharded_statevector_parameter_value_and_grad,
    run_jax_sharded_statevector,
)
from .statevector_gradient_records import (
    JAXShardedStatevectorParameterGradientResult,
)
from .statevector_records import (
    JAXShardedStatevectorResult,
    JAXShardedStatevectorTrainingPlan,
    JAXStatevectorShardState,
)
from .statevector_training import plan_jax_sharded_statevector_training

__all__ = (
    "JAXStatevectorShardState",
    "JAXShardedStatevectorResult",
    "JAXShardedStatevectorParameterGradientResult",
    "JAXShardedStatevectorTrainingPlan",
    "run_jax_sharded_statevector",
    "jax_sharded_statevector_parameter_value_and_grad",
    "plan_jax_sharded_statevector_training",
)
