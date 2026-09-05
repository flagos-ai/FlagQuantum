"""Compatibility exports for the decomposed optional JAX runtime.

Implementation ownership lives in semantically named modules. Importing this
shim preserves the documented compatibility objects without initializing JAX.
"""

from __future__ import annotations

from .backend_dispatch import plan_jax_distributed_quantum_backend
from .mps_execution import run_jax_sharded_mps
from .mps_gradient_result import JAXShardedMPSParameterGradientResult
from .mps_gradients import jax_sharded_mps_parameter_value_and_grad
from .mps_planning import (
    plan_jax_sharded_mps_parameter_flow,
    plan_jax_sharded_mps_training,
)
from .mps_result import JAXShardedMPSResult
from .mps_training_records import (
    JAXMPSRankShardState,
    JAXShardedMPSParameterFlowPlan,
    JAXShardedMPSParameterGateAssignment,
    JAXShardedMPSTrainingPlan,
)
from .planning_core import JAXDistributedQuantumPlan
from .runtime_environment import initialize_jax_distributed
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
from .tensor_network_execution import run_jax_sharded_tensor_network
from .tensor_network_gradients import (
    jax_sliced_tensor_network_parameter_value_and_grad,
    jax_sliced_tensor_network_value_and_grad,
)
from .tensor_network_records import (
    JAXShardedTensorNetworkResult,
    JAXSlicedTensorNetworkGradientResult,
    JAXSlicedTensorNetworkParameterGradientResult,
    JAXTensorNetworkNode,
    JAXTNSliceRankState,
)

__all__ = (
    "JAXDistributedQuantumPlan",
    "JAXMPSRankShardState",
    "JAXShardedStatevectorResult",
    "JAXShardedStatevectorParameterGradientResult",
    "JAXShardedStatevectorTrainingPlan",
    "JAXShardedMPSResult",
    "JAXShardedMPSParameterGradientResult",
    "JAXShardedMPSTrainingPlan",
    "JAXShardedMPSParameterFlowPlan",
    "JAXShardedMPSParameterGateAssignment",
    "JAXShardedTensorNetworkResult",
    "JAXStatevectorShardState",
    "JAXSlicedTensorNetworkGradientResult",
    "JAXSlicedTensorNetworkParameterGradientResult",
    "JAXTNSliceRankState",
    "JAXTensorNetworkNode",
    "initialize_jax_distributed",
    "jax_sharded_mps_parameter_value_and_grad",
    "jax_sharded_statevector_parameter_value_and_grad",
    "jax_sliced_tensor_network_parameter_value_and_grad",
    "jax_sliced_tensor_network_value_and_grad",
    "plan_jax_sharded_mps_parameter_flow",
    "plan_jax_sharded_mps_training",
    "plan_jax_sharded_statevector_training",
    "plan_jax_distributed_quantum_backend",
    "run_jax_sharded_mps",
    "run_jax_sharded_statevector",
    "run_jax_sharded_tensor_network",
)
