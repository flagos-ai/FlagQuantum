"""Unstable task-level distributed workflows.

Only user-invokable workflows are discoverable. Low-level shard records,
executors, kernel counters, and evidence objects remain implementation details.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LEGACY_NAMES = (
    "DistributedEvidenceContract",
    "DistributedTransportEvidence",
    "HybridParallelPlan",
    "JAXDistributedQuantumPlan",
    "JAXMPSRankShardState",
    "JAXStatevectorShardState",
    "JAXTNSliceRankState",
    "StatevectorCheckpointPolicy",
    "StatevectorShard",
    "StatevectorShardState",
    "TorchDistributedStatevectorGradientResult",
    "TorchDistributedStatevectorResult",
    "distributed_tensor_network_amplitude",
    "distributed_tensor_network_amplitudes",
    "distributed_tensor_network_expectation",
    "distributed_tensor_network_expectations",
    "execute_torch_distributed_mps_forward",
    "execute_torch_distributed_mps_reverse",
    "execute_torch_distributed_statevector",
    "execute_torch_distributed_statevector_reverse",
    "mps_site_kernel_stats",
    "reset_mps_site_kernel_stats",
    "site_sharded_z_zz_observations",
    "train_distributed_mps",
    "train_distributed_statevector",
)
_PUBLIC_NAMES = (
    "distributed_tensor_network_amplitude",
    "distributed_tensor_network_amplitudes",
    "distributed_tensor_network_expectation",
    "distributed_tensor_network_expectations",
    "train_distributed_mps",
    "train_distributed_statevector",
)
__all__ = _PUBLIC_NAMES

_API_EXPORTS = {
    "DistributedEvidenceContract",
    "DistributedTransportEvidence",
    "JAXDistributedQuantumPlan",
    "JAXMPSRankShardState",
    "JAXStatevectorShardState",
    "JAXTNSliceRankState",
    "StatevectorShard",
    "StatevectorShardState",
}
_STATEVECTOR_EXPORTS = {
    "StatevectorCheckpointPolicy",
    "TorchDistributedStatevectorGradientResult",
    "TorchDistributedStatevectorResult",
    "execute_torch_distributed_statevector",
    "execute_torch_distributed_statevector_reverse",
}


def __getattr__(name: str) -> Any:
    if name == "HybridParallelPlan":
        return getattr(import_module("flagquantum.runtime.parallel"), name)
    if name in {
        "distributed_tensor_network_amplitude",
        "distributed_tensor_network_amplitudes",
        "distributed_tensor_network_expectation",
        "distributed_tensor_network_expectations",
    }:
        return getattr(
            import_module("flagquantum.runtime.distributed.tensor_network_execution"),
            name,
        )
    if name in {"train_distributed_mps", "train_distributed_statevector"}:
        module = {
            "train_distributed_mps": "flagquantum.runtime.backends.mps",
            "train_distributed_statevector": "flagquantum.runtime.backends.statevector",
        }[name]
        return getattr(import_module(module), name)
    if name in _API_EXPORTS:
        return getattr(import_module("flagquantum.api"), name)
    if name in _STATEVECTOR_EXPORTS:
        return getattr(import_module("flagquantum.runtime.backends.statevector"), name)
    if name == "execute_torch_distributed_mps_forward":
        return getattr(import_module("flagquantum.runtime.backends.mps.forward"), name)
    if name in {
        "execute_torch_distributed_mps_reverse",
        "site_sharded_z_zz_observations",
    }:
        return getattr(import_module("flagquantum.runtime.backends.mps.reverse"), name)
    if name in {"reset_mps_site_kernel_stats", "mps_site_kernel_stats"}:
        target = {
            "reset_mps_site_kernel_stats": "reset_site_kernel_stats",
            "mps_site_kernel_stats": "site_kernel_stats",
        }[name]
        return getattr(
            import_module("flagquantum.runtime.backends.mps.site_kernels"), target
        )
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
