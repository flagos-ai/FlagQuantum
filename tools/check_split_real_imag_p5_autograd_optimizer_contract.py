#!/usr/bin/env python3
"""Validate the staged P5 autograd and optimizer boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "contracts"
    / "split-real-imag-statevector-p5-autograd-optimizer-contract.toml"
)


def load_toml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))


def contract_errors(contract: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "schema": "flagquantum_split_real_imag_statevector_p5_autograd_optimizer_contract_v1",
        "maturity": "experimental",
        "phase_status": "cpu_autograd_and_accelerator_double_single_sgd_reference",
        "base_executor": "split_real_imag_statevector_p4_device_double_single",
        "representation": "double_single_fp32_complex",
        "distribution_semantics": "single_device_fast_path",
        "runtime_default": False,
        "implementation_available": True,
        "public_api_available": True,
        "native_autograd_available": True,
        "optimizer_available": True,
        "end_to_end_double_single_gradient_claim_allowed": False,
        "convergence_claim_allowed": False,
        "hardware_certification": False,
        "scalability_claim_allowed": False,
        "performance_claim_allowed": False,
        "production_claim_allowed": False,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            errors.append(f"split real/imag P5 contract {name} must be {value!r}")

    precision = contract.get("precision_boundary", {})
    expected_precision = {
        "state_representation": "double_single_high_low",
        "explicit_parameter_shift_gradient": "double_single_high_low",
        "pytorch_leaf_dtype": "float32",
        "pytorch_tensor_grad_dtype": "float32",
        "pytorch_tensor_grad_word_count": 1,
        "optimizer_master_parameter": "double_single_high_low",
        "optimizer_master_gradient": "double_single_high_low_required",
        "silent_dtype_demotion_allowed": False,
        "single_tensor_grad_double_single_label_allowed": False,
    }
    for name, value in expected_precision.items():
        if precision.get(name) != value:
            errors.append(f"split real/imag P5 precision boundary {name} drifted")

    bridge = contract.get("autograd_bridge", {})
    if bridge.get("delivered_tensor_grad_precision") != "float32_boundary":
        errors.append("P5 must not label a single PyTorch tensor.grad Double-Single")
    for name in (
        "native_autograd_claim_requires_implementation",
        "native_autograd_claim_requires_gradcheck",
        "native_autograd_claim_requires_device_evidence",
    ):
        if bridge.get(name) is not True:
            errors.append(f"split real/imag P5 autograd claim gate {name} is required")
    expected_bridge_boundary = {
        "device_scope": "cpu_only",
        "higher_order_autograd": False,
        "gradcheck_dtype": "float32",
        "gradcheck_epsilon": 1e-3,
        "gradcheck_absolute_tolerance": 1e-4,
        "gradcheck_relative_tolerance": 1e-3,
    }
    for name, value in expected_bridge_boundary.items():
        if bridge.get(name) != value:
            errors.append(f"split real/imag P5 autograd boundary {name} drifted")

    optimizer = contract.get("precision_optimizer", {})
    if optimizer.get("initial_algorithm") != "double_single_sgd_without_momentum":
        errors.append("split real/imag P5 initial optimizer scope drifted")
    if optimizer.get("gradient_source") != (
        "explicit_double_single_parameter_shift_result"
    ):
        errors.append("P5 optimizer must consume an explicit Double-Single gradient")
    if optimizer.get("device_scope") != "single_device_cpu_cuda_flagos_reference":
        errors.append("P5 optimizer single-device reference scope drifted")
    if optimizer.get("implementation_available") is not True:
        errors.append("P5 optimizer implementation availability drifted")
    if optimizer.get("tensor_grad_used") is not False:
        errors.append("P5 optimizer must not consume Tensor.grad")
    if (
        optimizer.get("standard_torch_optimizer_compatibility_claim_allowed")
        is not False
    ):
        errors.append("P5 must not claim standard torch optimizer equivalence")
    if (
        optimizer.get("standard_tensor_grad_as_double_single_input_allowed")
        is not False
    ):
        errors.append("P5 must reject a one-word tensor.grad as Double-Single input")

    scope = contract.get("scope", {})
    if set(scope.get("parameterized_gates", ())) != {
        "rx",
        "ry",
        "rz",
        "rxx",
        "ryy",
        "rzz",
    }:
        errors.append("split real/imag P5 parameterized gate scope drifted")
    required_unsupported = {
        "standard_torch_optimizer_equivalence",
        "double_single_tensor_grad_claim",
        "adam",
        "adamw",
        "momentum",
        "higher_order_autograd",
        "accelerator_autograd",
        "compiled_autograd",
        "distributed_execution",
        "flagcx_collectives",
        "algorithmic_convergence_certification",
        "performance_claim",
        "production_claim",
    }
    if not required_unsupported <= set(scope.get("unsupported", ())):
        errors.append("split real/imag P5 fail-closed scope drifted")

    matrix = contract.get("acceptance_matrix", {})
    expected_matrix = {
        "depths": [8, 32, 128],
        "seeds": [0, 7],
        "optimizer_steps": [1, 16, 64],
        "workloads": ["two_qubit_vqe_trajectory", "three_qubit_qaoa_trajectory"],
        "devices": ["cpu", "single_cuda_reference", "single_flagos_reference"],
    }
    if matrix != expected_matrix:
        errors.append("split real/imag P5 acceptance matrix drifted")
    thresholds = contract.get("acceptance_thresholds", {})
    if not thresholds or any(float(value) <= 0.0 for value in thresholds.values()):
        errors.append("split real/imag P5 acceptance thresholds must be positive")
    if thresholds.get("max_optimizer_parameter_absolute_error") != 2e-8:
        errors.append("split real/imag P5 optimizer evidence threshold drifted")

    gates = contract.get("claim_gates", {})
    if gates.get("contract_only") is not False:
        errors.append("split real/imag P5 must report the implemented CPU slice")
    required_gates = {
        "cpu_reference_required",
        "finite_difference_diagnostic_required",
        "torch_gradcheck_required",
        "native_cuda_required",
        "torch_fl_flagos_required",
        "optimizer_trajectory_required",
        "algorithmic_convergence_required_for_convergence_claim",
        "flagcx_required_for_distributed_claim",
    }
    if any(gates.get(name) is not True for name in required_gates):
        errors.append("split real/imag P5 claim gates must remain fail-closed")
    if gates.get("cpu_optimizer_trajectory_evidence") is not True:
        errors.append("split real/imag P5 CPU optimizer evidence must be recorded")
    for name in (
        "native_cuda_optimizer_trajectory_evidence",
        "torch_fl_flagos_optimizer_trajectory_evidence",
    ):
        if gates.get(name) is not True:
            errors.append(f"split real/imag P5 optimizer evidence {name} is required")

    implementation = (
        ROOT / "flagquantum/runtime/executors/statevector/split_real_imag_autograd.py"
    )
    source = implementation.read_text(encoding="utf-8")
    required_source_tokens = {
        "torch.autograd.Function",
        "parameter_shift_split_real_imag_device_double_single_gradient",
        '"float32_boundary"',
        'resolved_device.type != "cpu"',
    }
    if any(token not in source for token in required_source_tokens):
        errors.append("split real/imag P5 CPU bridge implementation drifted")

    optimizer_implementation = (
        ROOT
        / "flagquantum/runtime/executors/statevector/split_real_imag_autograd_optimizer.py"
    )
    optimizer_source = optimizer_implementation.read_text(encoding="utf-8")
    required_optimizer_tokens = {
        "DoubleSingleTensor",
        "gradient.multiply",
        ".subtract(",
        ".renormalized()",
        '"tensor_grad_used": False',
        '"torch_optimizer_compatible": False',
    }
    if any(token not in optimizer_source for token in required_optimizer_tokens):
        errors.append("split real/imag P5 Double-Single SGD implementation drifted")

    for name, raw_path in contract.get("verification", {}).items():
        if not isinstance(raw_path, str) or not (ROOT / raw_path).is_file():
            errors.append(f"split real/imag P5 verification path {name!r} is missing")
    return tuple(errors)


def main() -> int:
    errors = contract_errors(load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag statevector P5 autograd/optimizer contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
