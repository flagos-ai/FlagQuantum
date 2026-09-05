"""Independent tensor-network facade over its authoritative implementation."""

from __future__ import annotations

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
    "JAXTensorNetworkNode",
    "JAXTNSliceRankState",
    "JAXShardedTensorNetworkResult",
    "JAXSlicedTensorNetworkGradientResult",
    "JAXSlicedTensorNetworkParameterGradientResult",
    "run_jax_sharded_tensor_network",
    "jax_sliced_tensor_network_value_and_grad",
    "jax_sliced_tensor_network_parameter_value_and_grad",
)
