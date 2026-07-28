"""Explicitly unstable FlagQuantum APIs with no compatibility guarantee."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "DistributedEvidenceContract",
    "DistributedTransportEvidence",
    "JAXDistributedQuantumPlan",
    "JAXMPSRankShardState",
    "JAXStatevectorShardState",
    "JAXTNSliceRankState",
    "StatevectorShard",
    "StatevectorShardState",
    "TorchDistributedStatevectorResult",
    "execute_torch_distributed_statevector",
    "StatevectorCheckpointPolicy",
    "TorchDistributedStatevectorGradientResult",
    "execute_torch_distributed_statevector_reverse",
    "execute_torch_distributed_mps_forward",
    "execute_torch_distributed_mps_reverse",
    "site_sharded_z_zz_observations",
    "reset_mps_site_kernel_stats",
    "mps_site_kernel_stats",
    "DynamicCircuit",
    "DynamicBackendCompatibility",
    "DynamicExecutionResult",
    "export_dynamic_qasm3",
    "export_braket_iqm_dynamic_qasm3",
    "export_dynamic_qasm3_for_backend",
    "run_dynamic",
    "assess_dynamic_backend",
    "create_dynamic_deployment_package",
    "deploy_dynamic_circuit",
    "route_dynamic_circuit",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name in {
        "TorchDistributedStatevectorResult",
        "execute_torch_distributed_statevector",
    }:
        return getattr(import_module("flagquantum.runtime.backends.statevector"), name)
    if name in {
        "StatevectorCheckpointPolicy",
        "TorchDistributedStatevectorGradientResult",
        "execute_torch_distributed_statevector_reverse",
    }:
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
    if name in {
        "DynamicCircuit",
        "DynamicBackendCompatibility",
        "DynamicExecutionResult",
        "export_dynamic_qasm3",
        "export_braket_iqm_dynamic_qasm3",
        "export_dynamic_qasm3_for_backend",
        "run_dynamic",
        "assess_dynamic_backend",
        "create_dynamic_deployment_package",
        "deploy_dynamic_circuit",
        "route_dynamic_circuit",
    }:
        return getattr(import_module("flagquantum.runtime.dynamic"), name)
    return getattr(import_module("flagquantum.api"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
