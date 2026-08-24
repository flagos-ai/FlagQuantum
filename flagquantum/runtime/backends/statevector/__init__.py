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
    "SplitRealImagPrecisionConformanceReport",
    "SplitRealImagPrecisionExpectationResult",
    "SplitRealImagPrecisionGradientResult",
    "SplitRealImagDoubleSingleConformanceReport",
    "SplitRealImagDoubleSingleExpectationResult",
    "SplitRealImagDoubleSingleGradientResult",
    "SplitRealImagDoubleSingleStatevectorResult",
    "execute_split_real_imag_expectation",
    "execute_split_real_imag_precision_expectation",
    "execute_split_real_imag_statevector",
    "parameter_shift_split_real_imag_gradient",
    "parameter_shift_split_real_imag_precision_gradient",
    "run_split_real_imag_conformance",
    "run_split_real_imag_training_conformance",
    "run_split_real_imag_precision_conformance",
    "split_real_imag_p2_accuracy_envelope",
    "split_real_imag_p2_precision_plan",
    "execute_split_real_imag_double_single_expectation",
    "execute_split_real_imag_double_single_statevector",
    "parameter_shift_split_real_imag_double_single_gradient",
    "run_split_real_imag_double_single_conformance",
    "split_real_imag_p3_accuracy_envelope",
    "split_real_imag_p3_precision_plan",
)

_EXPORTS = {
    "SplitRealImagDoubleSingleConformanceReport": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single_conformance",
        "SplitRealImagDoubleSingleConformanceReport",
    ),
    "SplitRealImagDoubleSingleExpectationResult": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single",
        "SplitRealImagDoubleSingleExpectationResult",
    ),
    "SplitRealImagDoubleSingleGradientResult": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single",
        "SplitRealImagDoubleSingleGradientResult",
    ),
    "SplitRealImagDoubleSingleStatevectorResult": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single",
        "SplitRealImagDoubleSingleStatevectorResult",
    ),
    "execute_split_real_imag_double_single_expectation": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single",
        "execute_split_real_imag_double_single_expectation",
    ),
    "execute_split_real_imag_double_single_statevector": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single",
        "execute_split_real_imag_double_single_statevector",
    ),
    "parameter_shift_split_real_imag_double_single_gradient": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single",
        "parameter_shift_split_real_imag_double_single_gradient",
    ),
    "run_split_real_imag_double_single_conformance": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single_conformance",
        "run_split_real_imag_double_single_conformance",
    ),
    "split_real_imag_p3_accuracy_envelope": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single",
        "split_real_imag_p3_accuracy_envelope",
    ),
    "split_real_imag_p3_precision_plan": (
        "flagquantum.runtime.backends.statevector.split_real_imag_double_single",
        "split_real_imag_p3_precision_plan",
    ),
    "SplitRealImagPrecisionConformanceReport": (
        "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "SplitRealImagPrecisionConformanceReport",
    ),
    "SplitRealImagPrecisionExpectationResult": (
        "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "SplitRealImagPrecisionExpectationResult",
    ),
    "SplitRealImagPrecisionGradientResult": (
        "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "SplitRealImagPrecisionGradientResult",
    ),
    "execute_split_real_imag_precision_expectation": (
        "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "execute_split_real_imag_precision_expectation",
    ),
    "parameter_shift_split_real_imag_precision_gradient": (
        "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "parameter_shift_split_real_imag_precision_gradient",
    ),
    "run_split_real_imag_precision_conformance": (
        "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "run_split_real_imag_precision_conformance",
    ),
    "split_real_imag_p2_accuracy_envelope": (
        "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "split_real_imag_p2_accuracy_envelope",
    ),
    "split_real_imag_p2_precision_plan": (
        "flagquantum.runtime.backends.statevector.split_real_imag_precision",
        "split_real_imag_p2_precision_plan",
    ),
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
