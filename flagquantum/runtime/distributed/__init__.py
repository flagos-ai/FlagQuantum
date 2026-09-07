"""Cycle-safe distributed orchestration contracts and topology APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "DistributedBackendPolicy",
    "DistributedExecutionRecord",
    "DistributedExecutionRequest",
    "DistributedExecutor",
    "DistributedIdentity",
    "DistributedIdentityError",
    "FlagOSWorkloadCapability",
    "FlagOSWorkloadCapabilityError",
    "FlagOSWorkloadCapabilityMatrix",
    "LocalTensor",
    "LocalTensorShard",
    "TorchDistributedContext",
    "destroy_torch_distributed",
    "distributed_backend_env_help",
    "init_torch_distributed",
    "resolve_distributed_backend_policy",
    "require_verified_flagcx",
    "torch_distributed_is_available",
    "build_flagos_workload_capability_matrix",
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
_IDENTITY_EXPORTS = {
    "DistributedIdentity",
    "DistributedIdentityError",
    "require_verified_flagcx",
}
_WORKLOAD_CAPABILITY_EXPORTS = {
    "FlagOSWorkloadCapability",
    "FlagOSWorkloadCapabilityError",
    "FlagOSWorkloadCapabilityMatrix",
    "build_flagos_workload_capability_matrix",
}
_CONTEXT_EXPORTS = {
    "TorchDistributedContext",
    "destroy_torch_distributed",
    "init_torch_distributed",
    "torch_distributed_is_available",
}


def __getattr__(name: str) -> Any:
    if name in _BACKEND_EXPORTS:
        module = import_module(".backend_policy", __name__)
    elif name in _PROTOCOL_EXPORTS:
        module = import_module("flagquantum.runtime.distributed.protocols")
    elif name in _IDENTITY_EXPORTS:
        module = import_module("flagquantum.runtime.distributed.identity")
    elif name in _WORKLOAD_CAPABILITY_EXPORTS:
        module = import_module("flagquantum.runtime.distributed.workload_capability")
    elif name in _CONTEXT_EXPORTS:
        module = import_module("flagquantum.runtime.distributed.context")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
