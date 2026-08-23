"""Cycle-safe distributed orchestration contracts and topology APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "DistributedBackendPolicy",
    "DistributedBoundaryProtocol",
    "DistributedBoundarySync",
    "DistributedExecutionRecord",
    "DistributedExecutionRequest",
    "DistributedExecutor",
    "DistributedShardPlan",
    "DistributedSliceTask",
    "DistributedTensorNetworkAmplitude",
    "DistributedTensorNetworkAmplitudes",
    "DistributedTensorNetworkExpectation",
    "DistributedTensorNetworkExpectations",
    "LocalTensor",
    "LocalTensorShard",
    "TorchDistributedContext",
    "destroy_torch_distributed",
    "distributed_backend_env_help",
    "init_torch_distributed",
    "clear_mps_static_descriptor_cache",
    "mps_p2p_stats",
    "mps_static_descriptor_cache_entries",
    "reset_mps_p2p_stats",
    "resolve_distributed_backend_policy",
    "run_distributed_mps",
    "distributed_tensor_network_amplitude",
    "distributed_tensor_network_amplitudes",
    "distributed_tensor_network_expectation",
    "distributed_tensor_network_expectations",
    "run_distributed_tensor_network",
    "torch_distributed_is_available",
    "warmup_mps_neighbor_communicators",
)

_BACKEND_EXPORTS = {
    "DistributedBackendPolicy",
    "LocalTensor",
    "LocalTensorShard",
    "distributed_backend_env_help",
    "resolve_distributed_backend_policy",
}
_PROTOCOL_EXPORTS = {
    "DistributedExecutionRecord",
    "DistributedExecutionRequest",
    "DistributedExecutor",
}
_MPS_TRANSPORT_EXPORTS = {
    "clear_mps_static_descriptor_cache",
    "mps_p2p_stats",
    "mps_static_descriptor_cache_entries",
    "reset_mps_p2p_stats",
    "warmup_mps_neighbor_communicators",
}
_ENGINE_EXPORTS = {
    "DistributedBoundaryProtocol",
    "DistributedBoundarySync",
    "DistributedShardPlan",
    "DistributedSliceTask",
    "DistributedTensorNetworkAmplitude",
    "DistributedTensorNetworkAmplitudes",
    "DistributedTensorNetworkExpectation",
    "DistributedTensorNetworkExpectations",
    "TorchDistributedContext",
    "destroy_torch_distributed",
    "init_torch_distributed",
    "run_distributed_mps",
    "distributed_tensor_network_amplitude",
    "distributed_tensor_network_amplitudes",
    "distributed_tensor_network_expectation",
    "distributed_tensor_network_expectations",
    "run_distributed_tensor_network",
    "torch_distributed_is_available",
}


def __getattr__(name: str) -> Any:
    if name in _BACKEND_EXPORTS:
        module = import_module(".backend_policy", __name__)
    elif name in _PROTOCOL_EXPORTS:
        module = import_module("flagquantum.runtime.distributed.protocols")
    elif name in _MPS_TRANSPORT_EXPORTS:
        module = import_module("flagquantum.runtime.distributed.mps_transport")
    elif name in _ENGINE_EXPORTS:
        module = import_module("flagquantum.runtime.distributed.engine")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
