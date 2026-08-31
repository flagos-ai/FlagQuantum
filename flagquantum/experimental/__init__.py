"""Explicitly unstable FlagQuantum APIs with no compatibility guarantee."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "distributed",
    "execution",
    "mps",
    "planning",
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
    "DynamicFeatureSet",
    "DynamicFeatureReport",
    "DynamicConformanceCase",
    "DynamicConformanceResult",
    "LOCAL_TRAJECTORY_FEATURES",
    "QISKIT_AER_DYNAMIC_FEATURES",
    "BRAKET_IQM_DYNAMIC_FEATURES",
    "assess_dynamic_features",
    "dynamic_conformance_cases",
    "run_dynamic_conformance",
    "run_qiskit_aer_dynamic",
    "run_qiskit_aer_qasm3_round_trip",
    "from_qiskit",
    "to_qiskit",
    "TEBDResult",
    "run_tebd",
    "run_double_single_conformance",
    "SplitRealImagConformanceReport",
    "SplitRealImagExpectationResult",
    "SplitRealImagParameterShiftResult",
    "SplitRealImagStatevectorResult",
    "SplitRealImagTrainingConformanceReport",
    "SplitRealImagPrecisionConformanceReport",
    "SplitRealImagPrecisionExpectationResult",
    "SplitRealImagPrecisionGradientResult",
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
    "SplitRealImagDoubleSingleConformanceReport",
    "SplitRealImagDoubleSingleExpectationResult",
    "SplitRealImagDoubleSingleGradientResult",
    "SplitRealImagDoubleSingleStatevectorResult",
    "execute_split_real_imag_double_single_expectation",
    "execute_split_real_imag_double_single_statevector",
    "parameter_shift_split_real_imag_double_single_gradient",
    "run_split_real_imag_double_single_conformance",
    "split_real_imag_p3_accuracy_envelope",
    "split_real_imag_p3_precision_plan",
    "SplitRealImagDeviceDoubleSingleConformanceReport",
    "SplitRealImagDeviceDoubleSingleExpectationResult",
    "SplitRealImagDeviceDoubleSingleGradientResult",
    "SplitRealImagDeviceDoubleSingleStatevectorResult",
    "execute_split_real_imag_device_double_single_expectation",
    "execute_split_real_imag_device_double_single_statevector",
    "parameter_shift_split_real_imag_device_double_single_gradient",
    "run_split_real_imag_device_double_single_conformance",
    "split_real_imag_p4_accuracy_envelope",
    "split_real_imag_p4_precision_plan",
    "SplitRealImagAutogradConformanceReport",
    "run_split_real_imag_autograd_conformance",
    "split_real_imag_device_double_single_autograd_expectation",
    "split_real_imag_p5_autograd_bridge_summary",
    "SplitRealImagDoubleSingleSGDState",
    "SplitRealImagDoubleSingleSGDStepResult",
    "SplitRealImagOptimizerConformanceReport",
    "double_single_sgd_step",
    "initialize_split_real_imag_double_single_sgd",
    "run_split_real_imag_optimizer_conformance",
    "split_real_imag_double_single_sgd_step",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name in {"distributed", "execution", "mps", "planning"}:
        return import_module(f"flagquantum.experimental.{name}")
    if name in {
        "TEBDResult",
        "run_tebd",
    }:
        return getattr(import_module("flagquantum.simulation.tebd"), name)
    if name == "run_double_single_conformance":
        return getattr(import_module("flagquantum.numerics.conformance"), name)
    if name in {
        "SplitRealImagOptimizerConformanceReport",
        "run_split_real_imag_optimizer_conformance",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_optimizer_conformance"
            ),
            name,
        )
    if name in {
        "SplitRealImagDoubleSingleSGDState",
        "SplitRealImagDoubleSingleSGDStepResult",
        "double_single_sgd_step",
        "initialize_split_real_imag_double_single_sgd",
        "split_real_imag_double_single_sgd_step",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_autograd_optimizer"
            ),
            name,
        )
    if name in {
        "SplitRealImagAutogradConformanceReport",
        "run_split_real_imag_autograd_conformance",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_autograd_conformance"
            ),
            name,
        )
    if name in {
        "split_real_imag_device_double_single_autograd_expectation",
        "split_real_imag_p5_autograd_bridge_summary",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_autograd"
            ),
            name,
        )
    if name in {
        "SplitRealImagDeviceDoubleSingleConformanceReport",
        "run_split_real_imag_device_double_single_conformance",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_device_double_single_conformance"
            ),
            name,
        )
    if name in {
        "SplitRealImagDeviceDoubleSingleExpectationResult",
        "SplitRealImagDeviceDoubleSingleGradientResult",
        "SplitRealImagDeviceDoubleSingleStatevectorResult",
        "execute_split_real_imag_device_double_single_expectation",
        "execute_split_real_imag_device_double_single_statevector",
        "parameter_shift_split_real_imag_device_double_single_gradient",
        "split_real_imag_p4_accuracy_envelope",
        "split_real_imag_p4_precision_plan",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_device_double_single"
            ),
            name,
        )
    if name in {
        "SplitRealImagDoubleSingleConformanceReport",
        "run_split_real_imag_double_single_conformance",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_double_single_conformance"
            ),
            name,
        )
    if name in {
        "SplitRealImagDoubleSingleExpectationResult",
        "SplitRealImagDoubleSingleGradientResult",
        "SplitRealImagDoubleSingleStatevectorResult",
        "execute_split_real_imag_double_single_expectation",
        "execute_split_real_imag_double_single_statevector",
        "parameter_shift_split_real_imag_double_single_gradient",
        "split_real_imag_p3_accuracy_envelope",
        "split_real_imag_p3_precision_plan",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_double_single"
            ),
            name,
        )
    if name in {
        "SplitRealImagPrecisionConformanceReport",
        "SplitRealImagPrecisionExpectationResult",
        "SplitRealImagPrecisionGradientResult",
        "execute_split_real_imag_precision_expectation",
        "parameter_shift_split_real_imag_precision_gradient",
        "run_split_real_imag_precision_conformance",
        "split_real_imag_p2_accuracy_envelope",
        "split_real_imag_p2_precision_plan",
    }:
        return getattr(
            import_module(
                "flagquantum.runtime.backends.statevector.split_real_imag_precision"
            ),
            name,
        )
    if name in {
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
    }:
        return getattr(
            import_module("flagquantum.runtime.backends.statevector.split_real_imag"),
            name,
        )
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
    if name in {
        "from_qiskit",
        "to_qiskit",
    }:
        return getattr(import_module("flagquantum.interop.qiskit"), name)
    if name in {
        "DynamicFeatureSet",
        "DynamicFeatureReport",
        "DynamicConformanceCase",
        "DynamicConformanceResult",
        "LOCAL_TRAJECTORY_FEATURES",
        "QISKIT_AER_DYNAMIC_FEATURES",
        "BRAKET_IQM_DYNAMIC_FEATURES",
        "assess_dynamic_features",
        "dynamic_conformance_cases",
        "run_dynamic_conformance",
        "run_qiskit_aer_dynamic",
        "run_qiskit_aer_qasm3_round_trip",
    }:
        return getattr(import_module("flagquantum.runtime.dynamic_conformance"), name)
    return getattr(import_module("flagquantum.api"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
