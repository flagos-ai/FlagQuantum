"""P4 device-generated gates with full Double-Single state evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch

from ....compute import resolve_platform_device
from ....core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ....core.numerics import (
    AccuracyMode,
    AccuracyRequirementContract,
    ComplexRepresentation,
    PrecisionPlanContract,
    RefinementStrategy,
)
from ....core.parameters import Parameter
from ....simulation.numerics.double_single import (
    DoubleSingleComplexTensor,
    DoubleSingleTensor,
    double_single_sum,
)
from ....simulation.statevector.double_single import (
    double_single_pauli_term_expectation,
    run_double_single_statevector,
)
from ....simulation.statevector.double_single_device_gates import (
    P4_PARAMETER_GATES,
    encode_device_double_single_matrix,
)
from .split_real_imag import (
    _canonical_parameter_bindings,
    _coerce_bounded_accuracy,
    _coerce_exact_precision_plan,
    _normalized_observables,
    _operator_profile_identity,
    _parameter_occurrences,
    _PauliTerm,
    _validate_p1_execution_scope,
)

P4_EXECUTOR = "split_real_imag_statevector_p4_device_double_single"
P4_PROFILE = "split_real_imag_statevector_p4_device_double_single"


def split_real_imag_p4_precision_plan() -> PrecisionPlanContract:
    return PrecisionPlanContract(
        complex_representation=ComplexRepresentation.DOUBLE_SINGLE_FP32.value,
        parameter_dtype="float32_or_double_single_device",
        gate_generation_dtype="double_single_fp32_device_trigonometry",
        state_storage_dtype="float32",
        kernel_compute_dtype="float32",
        reduction_dtype="double_single_fp32",
        decomposition_dtype="not_applicable",
        gradient_dtype="double_single_fp32",
        optimizer_master_dtype="not_integrated",
        communication_dtype="not_applicable",
        checkpoint_dtype="double_single_fp32",
        refinement=RefinementStrategy.ADAPTIVE_BLOCK_REPLAY.value,
        allow_dtype_demotion=False,
    )


def split_real_imag_p4_accuracy_envelope() -> AccuracyRequirementContract:
    return AccuracyRequirementContract(
        mode=AccuracyMode.ADAPTIVE.value,
        max_norm_drift=1e-9,
        max_expectation_abs_error=1e-9,
        max_expectation_rel_error=1e-8,
        max_gradient_rel_error=1e-8,
        min_gradient_cosine_similarity=0.99999999,
        max_state_infidelity=1e-9,
        require_determinism=False,
        require_convergence_evidence=False,
    )


def _coerce_plan(
    value: PrecisionPlanContract | Mapping[str, Any] | None,
) -> PrecisionPlanContract:
    return _coerce_exact_precision_plan(
        value,
        implemented=split_real_imag_p4_precision_plan(),
        error="P4 implements one device Double-Single plan",
    )


def _coerce_accuracy(
    value: AccuracyRequirementContract | Mapping[str, Any] | None,
) -> AccuracyRequirementContract:
    certified = split_real_imag_p4_accuracy_envelope()
    requested = _coerce_bounded_accuracy(
        value,
        certified=certified,
        maximum_fields=(
            "max_norm_drift",
            "max_expectation_abs_error",
            "max_expectation_rel_error",
            "max_gradient_rel_error",
            "max_state_infidelity",
        ),
        owner="P4",
    )
    if requested.require_determinism or requested.require_convergence_evidence:
        raise RuntimeError("P4 has no determinism or convergence certification")
    if any(
        getattr(requested, name) is not None
        for name in ("max_decomposition_residual", "max_truncation_error")
    ):
        raise RuntimeError("P4 does not implement decomposition or truncation")
    return requested


@dataclass(frozen=True)
class SplitRealImagDeviceDoubleSingleStatevectorResult:
    state: DoubleSingleComplexTensor
    circuit_hash: str
    gate_count: int
    normalization_count: int
    renormalize_every: int
    provider: str
    operator_profile: str
    operator_profile_hash: str
    operator_evidence_ids: tuple[str, ...]
    host_scalar_parameter_ingestion: bool
    executor: str = P4_EXECUTOR

    @property
    def device(self) -> torch.device:
        return self.state.real.high.device

    def cpu_complex128(self) -> torch.Tensor:
        return self.state.to("cpu").to_complex128().detach()

    def norm_cpu_float64(self) -> torch.Tensor:
        return (
            double_single_sum(self.state.abs_squared()).to("cpu").to_float64().detach()
        )

    def summary(self) -> dict[str, Any]:
        precision_plan = split_real_imag_p4_precision_plan()
        return {
            "schema": "flagquantum_split_real_imag_p4_statevector_result_v1",
            "executor": self.executor,
            "representation": "double_single_fp32_complex",
            "precision_mechanism": ComplexRepresentation.DOUBLE_SINGLE_FP32.value,
            "precision_class": "emulated_high_precision",
            "precision_plan": precision_plan.to_dict(),
            "precision_plan_hash": precision_plan.content_hash(),
            "native_complex128": False,
            "logical_complex128_certified": False,
            "automatic_runtime_selection": False,
            "distribution_semantics": "single_device_fast_path",
            "device": str(self.device),
            "provider": self.provider,
            "parameter_encoding": "float32_or_double_single_device",
            "gate_generation": "double_single_fp32_device_trigonometry",
            "device_only_double_single_trigonometry": True,
            "host_gate_encoding": False,
            "parameter_host_fallback": False,
            "state_host_fallback": False,
            "host_execution_fallback": False,
            "host_scalar_parameter_ingestion": self.host_scalar_parameter_ingestion,
            "host_sync_safety_checks": False,
            "device_async_safety_checks": True,
            "complex_accelerator_tensor_materialized": False,
            "state_word_dtype": "float32",
            "state_word_count_per_amplitude": 4,
            "gate_count": self.gate_count,
            "normalization_count": self.normalization_count,
            "renormalize_every": self.renormalize_every,
            "operator_profile": self.operator_profile,
            "operator_profile_hash": self.operator_profile_hash,
            "operator_evidence_ids": list(self.operator_evidence_ids),
            "convergence_certification": False,
            "hardware_certification": False,
            "scalability_claim_allowed": False,
            "performance_claim_allowed": False,
            "production_claim_allowed": False,
        }


@dataclass(frozen=True)
class SplitRealImagDeviceDoubleSingleExpectationResult:
    state: SplitRealImagDeviceDoubleSingleStatevectorResult
    value: DoubleSingleTensor
    term_expectations: DoubleSingleTensor
    term_count: int
    precision_plan: PrecisionPlanContract
    accuracy_requirement: AccuracyRequirementContract

    def cpu_float64(self) -> torch.Tensor:
        return self.value.to("cpu").to_float64().detach()

    def summary(self) -> dict[str, Any]:
        summary = self.state.summary()
        summary.update(
            {
                "schema": "flagquantum_split_real_imag_p4_expectation_result_v1",
                "term_count": self.term_count,
                "precision_plan": self.precision_plan.to_dict(),
                "precision_plan_hash": self.precision_plan.content_hash(),
                "accuracy_requirement": self.accuracy_requirement.to_dict(),
                "accuracy_requirement_hash": self.accuracy_requirement.content_hash(),
            }
        )
        return summary


@dataclass(frozen=True)
class SplitRealImagDeviceDoubleSingleGradientResult:
    expectation: SplitRealImagDeviceDoubleSingleExpectationResult
    parameter_order: tuple[str, ...]
    gradient: DoubleSingleTensor
    shifted_evaluations: int
    shift: float = math.pi / 2.0

    def cpu_float64(self) -> torch.Tensor:
        return self.gradient.to("cpu").to_float64().detach()

    def summary(self) -> dict[str, Any]:
        summary = self.expectation.summary()
        summary.update(
            {
                "schema": "flagquantum_split_real_imag_p4_gradient_result_v1",
                "gradient_method": "parameter_shift",
                "parameter_order": self.parameter_order,
                "shifted_evaluations": self.shifted_evaluations,
            }
        )
        return summary


def _validate_scope(ir: CircuitIR) -> None:
    for instruction in ir.instructions:
        if instruction.params and instruction.name not in P4_PARAMETER_GATES:
            raise NotImplementedError(
                f"P4 device gate generation does not support {instruction.name!r}"
            )


def _execute_p4_statevector(
    ir: CircuitIR,
    *,
    bindings: Mapping[str, Any],
    device: str | torch.device,
    preflight: bool,
    renormalize_every: int,
    shifted_occurrence: tuple[int, str, int] | None = None,
) -> SplitRealImagDeviceDoubleSingleStatevectorResult:
    resolved_device = resolve_platform_device(device)
    profile, profile_hash, evidence_ids, provider = _operator_profile_identity(
        device=resolved_device,
        profile_name=P4_PROFILE,
        preflight=preflight,
    )
    host_ingestion = False

    def encoded_gates() -> Iterator[tuple[DoubleSingleComplexTensor, Sequence[int]]]:
        nonlocal host_ingestion
        for index, instruction in enumerate(ir.instructions):
            shifted = shifted_occurrence is not None and shifted_occurrence[:2] == (
                index,
                "theta",
            )
            shift_direction = (
                shifted_occurrence[2] if shifted_occurrence is not None else 0
            )
            matrix, instruction_host_ingestion = encode_device_double_single_matrix(
                instruction,
                bindings=bindings,
                device=resolved_device,
                shift_parameter="theta" if shifted else None,
                shift_direction=shift_direction if shifted else 0,
            )
            host_ingestion = host_ingestion or instruction_host_ingestion
            yield matrix, instruction.wires

    state, normalization_count = run_double_single_statevector(
        encoded_gates(),
        n_wires=ir.n_wires,
        device=resolved_device,
        renormalize_every=renormalize_every,
    )
    if state.real.high.device.type != resolved_device.type:
        raise RuntimeError("P4 statevector escaped the requested logical device")
    return SplitRealImagDeviceDoubleSingleStatevectorResult(
        state=state,
        circuit_hash=ir.content_hash,
        gate_count=len(ir.instructions),
        normalization_count=normalization_count,
        renormalize_every=renormalize_every,
        provider=provider,
        operator_profile=profile,
        operator_profile_hash=profile_hash,
        operator_evidence_ids=evidence_ids,
        host_scalar_parameter_ingestion=host_ingestion,
    )


def execute_split_real_imag_device_double_single_statevector(
    circuit_or_ir: Any,
    *,
    device: str | torch.device = "cpu",
    preflight: bool = True,
    renormalize_every: int = 16,
) -> SplitRealImagDeviceDoubleSingleStatevectorResult:
    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    _validate_scope(ir)
    if _parameter_occurrences(ir):
        raise ValueError("bind all named parameters before P4 statevector execution")
    return _execute_p4_statevector(
        ir,
        bindings={},
        device=device,
        preflight=preflight,
        renormalize_every=renormalize_every,
    )


def _coefficient_pair(term: _PauliTerm, *, device: torch.device) -> DoubleSingleTensor:
    value = term.coefficient
    if isinstance(value, DoubleSingleTensor):
        if value.high.device != device or value.high.numel() != 1:
            raise ValueError("P4 Double-Single coefficients must reside on device")
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.is_complex():
            raise ValueError("P4 observable coefficients must be real scalars")
        if value.dtype != torch.float32:
            raise TypeError("P4 tensor coefficients must use float32")
        tensor = value.detach().to(device).reshape(())
    else:
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
    return DoubleSingleTensor.from_float32(tensor)


def _term_expectation(
    state: SplitRealImagDeviceDoubleSingleStatevectorResult,
    term: _PauliTerm,
    *,
    n_wires: int,
) -> DoubleSingleTensor:
    def matrix_for_op(wire: int, name: str) -> DoubleSingleComplexTensor:
        matrix, _ = encode_device_double_single_matrix(
            Instruction(name=name, wires=(wire,)),
            bindings={},
            device=state.device,
        )
        return matrix

    return double_single_pauli_term_expectation(
        state.state,
        term.ops,
        _coefficient_pair(term, device=state.device),
        n_wires=n_wires,
        matrix_for_op=matrix_for_op,
    )


def _expectation(
    ir: CircuitIR,
    terms: Sequence[_PauliTerm],
    *,
    bindings: Mapping[str, Any],
    device: str | torch.device,
    preflight: bool,
    renormalize_every: int,
    precision_plan: PrecisionPlanContract,
    accuracy_requirement: AccuracyRequirementContract,
    shifted_occurrence: tuple[int, str, int] | None = None,
) -> SplitRealImagDeviceDoubleSingleExpectationResult:
    state = _execute_p4_statevector(
        ir,
        bindings=bindings,
        device=device,
        preflight=preflight,
        renormalize_every=renormalize_every,
        shifted_occurrence=shifted_occurrence,
    )
    values = tuple(_term_expectation(state, term, n_wires=ir.n_wires) for term in terms)
    term_values = DoubleSingleTensor(
        torch.stack(tuple(value.high for value in values)),
        torch.stack(tuple(value.low for value in values)),
    )
    return SplitRealImagDeviceDoubleSingleExpectationResult(
        state=state,
        value=double_single_sum(term_values),
        term_expectations=term_values,
        term_count=len(terms),
        precision_plan=precision_plan,
        accuracy_requirement=accuracy_requirement,
    )


def execute_split_real_imag_device_double_single_expectation(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, Any] | None = None,
    device: str | torch.device = "cpu",
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None = None,
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None = None,
    preflight: bool = True,
    renormalize_every: int = 16,
) -> SplitRealImagDeviceDoubleSingleExpectationResult:
    plan = _coerce_plan(precision_plan)
    requirement = _coerce_accuracy(accuracy_requirement)
    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    _validate_scope(ir)
    occurrences = _parameter_occurrences(ir)
    bindings = _canonical_parameter_bindings(parameter_bindings or {})
    if set(bindings) != set(occurrences):
        raise ValueError(
            "P4 bindings must exactly match named parameters; "
            f"expected {sorted(occurrences)}, got {sorted(bindings)}"
        )
    return _expectation(
        ir,
        _normalized_observables(ir, observable),
        bindings=bindings,
        device=device,
        preflight=preflight,
        renormalize_every=renormalize_every,
        precision_plan=plan,
        accuracy_requirement=requirement,
    )


def parameter_shift_split_real_imag_device_double_single_gradient(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, Any],
    device: str | torch.device = "cpu",
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None = None,
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None = None,
    preflight: bool = True,
    renormalize_every: int = 16,
) -> SplitRealImagDeviceDoubleSingleGradientResult:
    plan = _coerce_plan(precision_plan)
    requirement = _coerce_accuracy(accuracy_requirement)
    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    _validate_scope(ir)
    occurrences = _parameter_occurrences(ir)
    bindings = _canonical_parameter_bindings(parameter_bindings)
    if set(bindings) != set(occurrences) or not occurrences:
        raise ValueError(
            "P4 bindings must exactly match at least one named parameter; "
            f"expected {sorted(occurrences)}, got {sorted(bindings)}"
        )
    terms = _normalized_observables(ir, observable)
    base = _expectation(
        ir,
        terms,
        bindings=bindings,
        device=device,
        preflight=preflight,
        renormalize_every=renormalize_every,
        precision_plan=plan,
        accuracy_requirement=requirement,
    )
    half = DoubleSingleTensor.from_float32(
        torch.tensor(0.5, dtype=torch.float32, device=base.state.device)
    )
    gradients = []
    shifted_evaluations = 0
    for name in sorted(occurrences):
        gradient = DoubleSingleTensor.zeros_like(half.high)
        for instruction_index, parameter_name in occurrences[name]:
            plus = _expectation(
                ir,
                terms,
                bindings=bindings,
                device=device,
                preflight=False,
                renormalize_every=renormalize_every,
                precision_plan=plan,
                accuracy_requirement=requirement,
                shifted_occurrence=(instruction_index, parameter_name, 1),
            ).value
            minus = _expectation(
                ir,
                terms,
                bindings=bindings,
                device=device,
                preflight=False,
                renormalize_every=renormalize_every,
                precision_plan=plan,
                accuracy_requirement=requirement,
                shifted_occurrence=(instruction_index, parameter_name, -1),
            ).value
            gradient = gradient.add(plus.subtract(minus).multiply(half))
            shifted_evaluations += 2
        gradients.append(gradient)
    return SplitRealImagDeviceDoubleSingleGradientResult(
        expectation=base,
        parameter_order=tuple(sorted(occurrences)),
        gradient=DoubleSingleTensor(
            torch.stack(tuple(value.high for value in gradients)),
            torch.stack(tuple(value.low for value in gradients)),
        ),
        shifted_evaluations=shifted_evaluations,
    )


__all__ = (
    "SplitRealImagDeviceDoubleSingleExpectationResult",
    "SplitRealImagDeviceDoubleSingleGradientResult",
    "SplitRealImagDeviceDoubleSingleStatevectorResult",
    "execute_split_real_imag_device_double_single_expectation",
    "execute_split_real_imag_device_double_single_statevector",
    "parameter_shift_split_real_imag_device_double_single_gradient",
    "split_real_imag_p4_accuracy_envelope",
    "split_real_imag_p4_precision_plan",
)
