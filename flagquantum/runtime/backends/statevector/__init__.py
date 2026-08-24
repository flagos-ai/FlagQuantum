"""Statevector backend boundary."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "BatchedStatevectorTrajectoryResult",
    "merge_noisy_statevector_results",
    "execute_torch_distributed_statevector",
    "execute_torch_distributed_statevector_reverse",
    "initialize_statevector_shard",
    "plan_distributed_statevector",
    "simulate_distributed_statevector_local",
    "StatevectorCheckpointPolicy",
    "TorchDistributedStatevectorGradientResult",
    "TorchDistributedStatevectorResult",
    "train_distributed_statevector",
    "run_noisy_statevector",
    "SplitRealImagConformanceReport",
    "SplitRealImagExpectationResult",
    "SplitRealImagParameterShiftResult",
    "SplitRealImagStatevectorResult",
    "SplitRealImagTrainingConformanceReport",
    "execute_split_real_imag_expectation",
    "execute_split_real_imag_statevector",
    "parameter_shift_split_real_imag_gradient",
    "run_split_real_imag_conformance",
    "run_split_real_imag_training_conformance",
)

_EXPORTS = {
    "SplitRealImagConformanceReport": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "SplitRealImagConformanceReport",
    ),
    "SplitRealImagStatevectorResult": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "SplitRealImagStatevectorResult",
    ),
    "SplitRealImagExpectationResult": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "SplitRealImagExpectationResult",
    ),
    "SplitRealImagParameterShiftResult": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "SplitRealImagParameterShiftResult",
    ),
    "SplitRealImagTrainingConformanceReport": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "SplitRealImagTrainingConformanceReport",
    ),
    "execute_split_real_imag_expectation": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "execute_split_real_imag_expectation",
    ),
    "execute_split_real_imag_statevector": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "execute_split_real_imag_statevector",
    ),
    "run_split_real_imag_conformance": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "run_split_real_imag_conformance",
    ),
    "parameter_shift_split_real_imag_gradient": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "parameter_shift_split_real_imag_gradient",
    ),
    "run_split_real_imag_training_conformance": (
        "flagquantum.runtime.backends.statevector.split_real_imag",
        "run_split_real_imag_training_conformance",
    ),
    "BatchedStatevectorTrajectoryResult": (
        "flagquantum.runtime.backends.statevector.noisy",
        "BatchedStatevectorTrajectoryResult",
    ),
    "run_noisy_statevector": (
        "flagquantum.runtime.backends.statevector.noisy",
        "run_noisy_statevector",
    ),
    "merge_noisy_statevector_results": (
        "flagquantum.runtime.backends.statevector.noisy",
        "merge_noisy_statevector_results",
    ),
    "TorchDistributedStatevectorResult": (
        "flagquantum.runtime.backends.statevector.forward",
        "TorchDistributedStatevectorResult",
    ),
    "execute_torch_distributed_statevector": (
        "flagquantum.runtime.backends.statevector.forward",
        "execute_torch_distributed_statevector",
    ),
    "initialize_statevector_shard": (
        "flagquantum.runtime.backends.statevector.forward",
        "initialize_statevector_shard",
    ),
    "plan_distributed_statevector": (
        "flagquantum.runtime.backends.statevector.state",
        "plan_distributed_statevector",
    ),
    "simulate_distributed_statevector_local": (
        "flagquantum.runtime.backends.statevector.state",
        "simulate_distributed_statevector_local",
    ),
    "StatevectorCheckpointPolicy": (
        "flagquantum.runtime.backends.statevector.reverse",
        "StatevectorCheckpointPolicy",
    ),
    "TorchDistributedStatevectorGradientResult": (
        "flagquantum.runtime.backends.statevector.reverse",
        "TorchDistributedStatevectorGradientResult",
    ),
    "execute_torch_distributed_statevector_reverse": (
        "flagquantum.runtime.backends.statevector.reverse",
        "execute_torch_distributed_statevector_reverse",
    ),
    "train_distributed_statevector": (
        "flagquantum.runtime.backends.statevector.training",
        "train_distributed_statevector",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), symbol)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
