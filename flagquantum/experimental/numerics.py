"""Unstable numerical representations and conformance interfaces."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "flagquantum.numerics.conformance": {"run_double_single_conformance"},
    "flagquantum.runtime.backends.statevector.split_real_imag": {
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
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_precision": {
        "SplitRealImagPrecisionConformanceReport",
        "SplitRealImagPrecisionExpectationResult",
        "SplitRealImagPrecisionGradientResult",
        "execute_split_real_imag_precision_expectation",
        "parameter_shift_split_real_imag_precision_gradient",
        "run_split_real_imag_precision_conformance",
        "split_real_imag_p2_accuracy_envelope",
        "split_real_imag_p2_precision_plan",
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_double_single": {
        "SplitRealImagDoubleSingleExpectationResult",
        "SplitRealImagDoubleSingleGradientResult",
        "SplitRealImagDoubleSingleStatevectorResult",
        "execute_split_real_imag_double_single_expectation",
        "execute_split_real_imag_double_single_statevector",
        "parameter_shift_split_real_imag_double_single_gradient",
        "split_real_imag_p3_accuracy_envelope",
        "split_real_imag_p3_precision_plan",
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_double_single_conformance": {
        "SplitRealImagDoubleSingleConformanceReport",
        "run_split_real_imag_double_single_conformance",
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_device_double_single": {
        "SplitRealImagDeviceDoubleSingleExpectationResult",
        "SplitRealImagDeviceDoubleSingleGradientResult",
        "SplitRealImagDeviceDoubleSingleStatevectorResult",
        "execute_split_real_imag_device_double_single_expectation",
        "execute_split_real_imag_device_double_single_statevector",
        "parameter_shift_split_real_imag_device_double_single_gradient",
        "split_real_imag_p4_accuracy_envelope",
        "split_real_imag_p4_precision_plan",
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_device_double_single_conformance": {
        "SplitRealImagDeviceDoubleSingleConformanceReport",
        "run_split_real_imag_device_double_single_conformance",
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_autograd": {
        "split_real_imag_device_double_single_autograd_expectation",
        "split_real_imag_p5_autograd_bridge_summary",
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_autograd_conformance": {
        "SplitRealImagAutogradConformanceReport",
        "run_split_real_imag_autograd_conformance",
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_autograd_optimizer": {
        "SplitRealImagDoubleSingleSGDState",
        "SplitRealImagDoubleSingleSGDStepResult",
        "double_single_sgd_step",
        "initialize_split_real_imag_double_single_sgd",
        "split_real_imag_double_single_sgd_step",
    },
    "flagquantum.runtime.backends.statevector.split_real_imag_optimizer_conformance": {
        "SplitRealImagOptimizerConformanceReport",
        "run_split_real_imag_optimizer_conformance",
    },
}

_NAME_TO_MODULE = {
    name: module for module, names in _EXPORT_MODULES.items() for name in names
}
__all__ = tuple(sorted(_NAME_TO_MODULE))


def __getattr__(name: str) -> Any:
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
