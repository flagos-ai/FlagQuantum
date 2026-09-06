"""Full Double-Single state evolution for FP32-only local devices.

P3 encodes complex128 CPU gate matrices into four FP32 words, then performs
state storage, gate application, normalization, observable reduction, and
parameter-shift accumulation with Double-Single arithmetic on the requested
device.  Host gate encoding is explicit metadata; state evolution never falls
back to the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from ....core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ....core.numerics import (
    AccuracyMode,
    AccuracyRequirementContract,
    ComplexRepresentation,
    PrecisionPlanContract,
    RefinementStrategy,
)
from ....core.parameters import (
    Parameter,
    ParameterExpression,
    bind_parameter_value,
)
from ....numerics.double_single import (
    DoubleSingleComplexTensor,
    DoubleSingleTensor,
    double_single_sum,
)
from ....providers.platform import get_platform_runtime, resolve_platform_device
from ....simulation.double_single_host_gates import encode_host_double_single_matrix
from ....simulation.double_single_statevector import (
    double_single_pauli_term_expectation,
    run_double_single_statevector,
)
from .split_real_imag import (
    _canonical_parameter_bindings,
    _normalized_observables,
    _parameter_occurrences,
    _PauliTerm,
    _validate_p1_execution_scope,
)

P3_EXECUTOR = "split_real_imag_statevector_p3_double_single"
P3_PROFILE = "split_real_imag_statevector_p3_double_single"


def split_real_imag_p3_precision_plan() -> PrecisionPlanContract:
    """Return the explicit precision plan implemented by P3."""

    return PrecisionPlanContract(
        complex_representation=ComplexRepresentation.DOUBLE_SINGLE_FP32.value,
        parameter_dtype="float64_host_encoded",
        gate_generation_dtype="complex128_host_encoded_to_double_single_fp32",
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


def split_real_imag_p3_accuracy_envelope() -> AccuracyRequirementContract:
    """Return the bounded accuracy envelope tested by P3 conformance."""

    return AccuracyRequirementContract(
        mode=AccuracyMode.ADAPTIVE.value,
        max_norm_drift=1e-10,
        max_expectation_abs_error=1e-10,
        max_expectation_rel_error=1e-9,
        max_gradient_rel_error=1e-8,
        min_gradient_cosine_similarity=0.999999999,
        max_state_infidelity=1e-10,
        require_determinism=False,
        require_convergence_evidence=False,
    )


def _coerce_plan(
    value: PrecisionPlanContract | Mapping[str, Any] | None,
) -> PrecisionPlanContract:
    requested = (
        split_real_imag_p3_precision_plan()
        if value is None
        else (
            value
            if isinstance(value, PrecisionPlanContract)
            else PrecisionPlanContract.from_dict(value)
        )
    )
    if requested != split_real_imag_p3_precision_plan():
        raise NotImplementedError(
            "split real/imag P3 implements one full Double-Single state plan; "
            "the requested precision plan is not executable"
        )
    return requested


def _coerce_accuracy(
    value: AccuracyRequirementContract | Mapping[str, Any] | None,
) -> AccuracyRequirementContract:
    requested = (
        split_real_imag_p3_accuracy_envelope()
        if value is None
        else (
            value
            if isinstance(value, AccuracyRequirementContract)
            else AccuracyRequirementContract.from_dict(value)
        )
    )
    certified = split_real_imag_p3_accuracy_envelope()
    for name in (
        "max_norm_drift",
        "max_expectation_abs_error",
        "max_expectation_rel_error",
        "max_gradient_rel_error",
        "max_state_infidelity",
    ):
        limit = getattr(requested, name)
        certified_limit = getattr(certified, name)
        if (
            limit is not None
            and certified_limit is not None
            and limit < certified_limit
        ):
            raise RuntimeError(
                f"split real/imag P3 cannot certify requested {name}={limit}; "
                f"certified envelope is {certified_limit}"
            )
    cosine = requested.min_gradient_cosine_similarity
    if (
        cosine is not None
        and certified.min_gradient_cosine_similarity is not None
        and cosine > certified.min_gradient_cosine_similarity
    ):
        raise RuntimeError(
            "split real/imag P3 cannot certify the requested gradient cosine"
        )
    unsupported = {
        name: getattr(requested, name)
        for name in ("max_decomposition_residual", "max_truncation_error")
        if getattr(requested, name) is not None
    }
    if requested.require_determinism:
        unsupported["require_determinism"] = True
    if requested.require_convergence_evidence:
        unsupported["require_convergence_evidence"] = True
    if unsupported:
        raise RuntimeError(
            "split real/imag P3 lacks evidence for requested constraints: "
            f"{unsupported}"
        )
    return requested


@dataclass(frozen=True)
class SplitRealImagDoubleSingleStatevectorResult:
    """One complex state represented by four device-resident FP32 words."""

    state: DoubleSingleComplexTensor
    circuit_hash: str
    gate_count: int
    normalization_count: int
    renormalize_every: int
    provider: str
    operator_profile: str
    operator_profile_hash: str
    operator_evidence_ids: tuple[str, ...]
    executor: str = P3_EXECUTOR

    @property
    def device(self) -> torch.device:
        return self.state.real.high.device

    @property
    def amplitude_count(self) -> int:
        return int(self.state.real.high.numel())

    def cpu_complex128(self) -> torch.Tensor:
        """Reconstruct the state only after transferring all words to CPU."""

        return self.state.to("cpu").to_complex128().detach()

    def norm_cpu_float64(self) -> torch.Tensor:
        return (
            double_single_sum(self.state.abs_squared()).to("cpu").to_float64().detach()
        )

    def summary(self) -> dict[str, Any]:
        word = self.state.real.high
        return {
            "schema": "flagquantum_split_real_imag_p3_statevector_result_v1",
            "executor": self.executor,
            "representation": "double_single_fp32_complex",
            "distribution_semantics": "single_device_fast_path",
            "device": str(self.device),
            "device_type": self.device.type,
            "provider": self.provider,
            "parameter_encoding": "float64_cpu_to_high_low_fp32",
            "gate_generation": "complex128_cpu_to_high_low_fp32",
            "host_gate_encoding": True,
            "state_host_fallback": False,
            "state_word_dtype": "float32",
            "state_word_count_per_amplitude": 4,
            "kernel_compute_dtype": "float32",
            "reduction_dtype": "double_single_fp32",
            "normalization_dtype": "double_single_fp32",
            "normalization_count": self.normalization_count,
            "renormalize_every": self.renormalize_every,
            "complex_accelerator_tensor_materialized": False,
            "native_autograd_supported": False,
            "distributed_supported": False,
            "scalability_claim_allowed": False,
            "runtime_default": False,
            "convergence_evidence": False,
            "provider_internal_route_audited": False,
            "circuit_hash": self.circuit_hash,
            "gate_count": self.gate_count,
            "amplitude_count": self.amplitude_count,
            "state_bytes": self.amplitude_count * word.element_size() * 4,
            "operator_profile": self.operator_profile,
            "operator_profile_hash": self.operator_profile_hash,
            "operator_evidence_ids": self.operator_evidence_ids,
        }


@dataclass(frozen=True)
class SplitRealImagDoubleSingleExpectationResult:
    state: SplitRealImagDoubleSingleStatevectorResult
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
                "schema": "flagquantum_split_real_imag_p3_expectation_result_v1",
                "term_count": self.term_count,
                "precision_plan": self.precision_plan.to_dict(),
                "precision_plan_hash": self.precision_plan.content_hash(),
                "accuracy_requirement": self.accuracy_requirement.to_dict(),
                "accuracy_requirement_hash": self.accuracy_requirement.content_hash(),
            }
        )
        return summary


@dataclass(frozen=True)
class SplitRealImagDoubleSingleGradientResult:
    expectation: SplitRealImagDoubleSingleExpectationResult
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
                "schema": "flagquantum_split_real_imag_p3_gradient_result_v1",
                "gradient_method": "parameter_shift",
                "parameter_order": self.parameter_order,
                "shifted_evaluations": self.shifted_evaluations,
            }
        )
        return summary


def _normalized_p3_bindings(
    parameter_bindings: Mapping[str | Parameter, Any],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for name, value in _canonical_parameter_bindings(parameter_bindings).items():
        tensor = (
            value.detach()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=torch.float64, device="cpu")
        )
        if tensor.numel() != 1:
            raise ValueError("split real/imag P3 requires scalar parameter bindings")
        if tensor.is_complex():
            if bool(torch.any(tensor.imag != 0).item()):
                raise ValueError("split real/imag P3 gate parameters must be real")
            tensor = tensor.real
        normalized[name] = tensor.detach().cpu().to(torch.float64).reshape(())
    return normalized


def _bind_p3_ir(
    ir: CircuitIR,
    bindings: Mapping[str, torch.Tensor],
    *,
    shifted_occurrence: tuple[int, str, float] | None = None,
) -> CircuitIR:
    instructions = []
    for index, instruction in enumerate(ir.instructions):
        params: dict[str, Any] = {}
        for parameter_name, value in instruction.params.items():
            bound = (
                bind_parameter_value(value, bindings)
                if isinstance(value, (Parameter, ParameterExpression))
                else value
            )
            tensor = (
                bound.detach()
                if isinstance(bound, torch.Tensor)
                else torch.as_tensor(bound, dtype=torch.float64, device="cpu")
            )
            if tensor.numel() != 1:
                raise ValueError("split real/imag P3 gate parameters must be scalar")
            if tensor.is_complex():
                if bool(torch.any(tensor.imag != 0).item()):
                    raise ValueError("split real/imag P3 gate parameters must be real")
                tensor = tensor.real
            encoded = tensor.detach().cpu().to(torch.float64).reshape(())
            if (
                shifted_occurrence is not None
                and (index, parameter_name) == shifted_occurrence[:2]
            ):
                encoded = encoded + shifted_occurrence[2]
            params[parameter_name] = encoded
        instructions.append(replace(instruction, params=params))
    return replace(
        ir, instructions=tuple(instructions), observables=(), measurements=()
    )


def _profile_identity(
    *, device: torch.device, preflight: bool
) -> tuple[str, str, tuple[str, ...], str]:
    platform = get_platform_runtime(device.type)
    identity = platform.identity()
    if preflight:
        from ...operator_probes import preflight_split_real_imag_statevector_p3

        report = preflight_split_real_imag_statevector_p3(
            device=device, provider=identity.provider
        )
        report.require_supported()
        return (
            report.profile,
            report.profile_hash,
            tuple(report.evidence_ids),
            identity.provider,
        )
    from ...capabilities import load_operator_profile

    profile = load_operator_profile(P3_PROFILE)
    return profile.name, profile.profile_hash, (), identity.provider


def _execute_bound_p3_statevector(
    ir: CircuitIR,
    *,
    device: str | torch.device,
    preflight: bool,
    renormalize_every: int,
) -> SplitRealImagDoubleSingleStatevectorResult:
    resolved_device = resolve_platform_device(device)
    profile, profile_hash, evidence_ids, provider = _profile_identity(
        device=resolved_device, preflight=preflight
    )
    encoded_gates = (
        (
            encode_host_double_single_matrix(instruction, device=resolved_device),
            instruction.wires,
        )
        for instruction in ir.instructions
    )
    state, normalization_count = run_double_single_statevector(
        encoded_gates,
        n_wires=ir.n_wires,
        device=resolved_device,
        renormalize_every=renormalize_every,
    )
    if state.real.high.device.type != resolved_device.type:
        raise RuntimeError("P3 statevector escaped the requested logical device")
    return SplitRealImagDoubleSingleStatevectorResult(
        state=state,
        circuit_hash=ir.content_hash,
        gate_count=len(ir.instructions),
        normalization_count=normalization_count,
        renormalize_every=renormalize_every,
        provider=provider,
        operator_profile=profile,
        operator_profile_hash=profile_hash,
        operator_evidence_ids=evidence_ids,
    )


def execute_split_real_imag_double_single_statevector(
    circuit_or_ir: Any,
    *,
    device: str | torch.device = "cpu",
    preflight: bool = True,
    renormalize_every: int = 16,
) -> SplitRealImagDoubleSingleStatevectorResult:
    """Execute a bounded local circuit with four-word Double-Single amplitudes."""

    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    if _parameter_occurrences(ir):
        raise ValueError("bind all named parameters before P3 statevector execution")
    return _execute_bound_p3_statevector(
        _bind_p3_ir(ir, {}),
        device=device,
        preflight=preflight,
        renormalize_every=renormalize_every,
    )


def _coefficient_pair(term: _PauliTerm, *, device: torch.device) -> DoubleSingleTensor:
    tensor = (
        term.coefficient.detach()
        if isinstance(term.coefficient, torch.Tensor)
        else torch.as_tensor(term.coefficient, dtype=torch.float64, device="cpu")
    )
    if tensor.is_complex():
        tensor = tensor.real
    return DoubleSingleTensor.from_float64(
        tensor.detach().cpu().to(torch.float64).reshape(())
    ).to(device)


def _term_expectation(
    state: SplitRealImagDoubleSingleStatevectorResult,
    term: _PauliTerm,
    *,
    n_wires: int,
) -> DoubleSingleTensor:
    return double_single_pauli_term_expectation(
        state.state,
        term.ops,
        _coefficient_pair(term, device=state.device),
        n_wires=n_wires,
        matrix_for_op=lambda wire, name: encode_host_double_single_matrix(
            Instruction(name=name, wires=(wire,)), device=state.device
        ),
    )


def _expectation_from_bound_p3_ir(
    ir: CircuitIR,
    terms: Sequence[_PauliTerm],
    *,
    device: str | torch.device,
    preflight: bool,
    renormalize_every: int,
    precision_plan: PrecisionPlanContract,
    accuracy_requirement: AccuracyRequirementContract,
) -> SplitRealImagDoubleSingleExpectationResult:
    state = _execute_bound_p3_statevector(
        ir,
        device=device,
        preflight=preflight,
        renormalize_every=renormalize_every,
    )
    values = tuple(_term_expectation(state, term, n_wires=ir.n_wires) for term in terms)
    term_values = DoubleSingleTensor(
        torch.stack(tuple(value.high for value in values)),
        torch.stack(tuple(value.low for value in values)),
    )
    return SplitRealImagDoubleSingleExpectationResult(
        state=state,
        value=double_single_sum(term_values),
        term_expectations=term_values,
        term_count=len(terms),
        precision_plan=precision_plan,
        accuracy_requirement=accuracy_requirement,
    )


def execute_split_real_imag_double_single_expectation(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, Any] | None = None,
    device: str | torch.device = "cpu",
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None = None,
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None = None,
    preflight: bool = True,
    renormalize_every: int = 16,
) -> SplitRealImagDoubleSingleExpectationResult:
    plan = _coerce_plan(precision_plan)
    requirement = _coerce_accuracy(accuracy_requirement)
    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    occurrences = _parameter_occurrences(ir)
    bindings = _normalized_p3_bindings(parameter_bindings or {})
    if set(bindings) != set(occurrences):
        raise ValueError(
            "parameter bindings must exactly match named circuit parameters; "
            f"expected {sorted(occurrences)}, got {sorted(bindings)}"
        )
    terms = _normalized_observables(ir, observable)
    return _expectation_from_bound_p3_ir(
        _bind_p3_ir(ir, bindings),
        terms,
        device=device,
        preflight=preflight,
        renormalize_every=renormalize_every,
        precision_plan=plan,
        accuracy_requirement=requirement,
    )


def parameter_shift_split_real_imag_double_single_gradient(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, Any],
    device: str | torch.device = "cpu",
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None = None,
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None = None,
    preflight: bool = True,
    renormalize_every: int = 16,
) -> SplitRealImagDoubleSingleGradientResult:
    plan = _coerce_plan(precision_plan)
    requirement = _coerce_accuracy(accuracy_requirement)
    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    occurrences = _parameter_occurrences(ir)
    bindings = _normalized_p3_bindings(parameter_bindings)
    if set(bindings) != set(occurrences) or not occurrences:
        raise ValueError(
            "parameter bindings must exactly match at least one named circuit "
            f"parameter; expected {sorted(occurrences)}, got {sorted(bindings)}"
        )
    terms = _normalized_observables(ir, observable)
    base = _expectation_from_bound_p3_ir(
        _bind_p3_ir(ir, bindings),
        terms,
        device=device,
        preflight=preflight,
        renormalize_every=renormalize_every,
        precision_plan=plan,
        accuracy_requirement=requirement,
    )
    gradients = []
    half = DoubleSingleTensor.from_float32(
        torch.tensor(0.5, dtype=torch.float32, device=base.state.device)
    )
    shifted_evaluations = 0
    for name in sorted(occurrences):
        gradient = DoubleSingleTensor.zeros_like(half.high)
        for instruction_index, parameter_name in occurrences[name]:
            plus = _expectation_from_bound_p3_ir(
                _bind_p3_ir(
                    ir,
                    bindings,
                    shifted_occurrence=(
                        instruction_index,
                        parameter_name,
                        math.pi / 2.0,
                    ),
                ),
                terms,
                device=device,
                preflight=False,
                renormalize_every=renormalize_every,
                precision_plan=plan,
                accuracy_requirement=requirement,
            ).value
            minus = _expectation_from_bound_p3_ir(
                _bind_p3_ir(
                    ir,
                    bindings,
                    shifted_occurrence=(
                        instruction_index,
                        parameter_name,
                        -math.pi / 2.0,
                    ),
                ),
                terms,
                device=device,
                preflight=False,
                renormalize_every=renormalize_every,
                precision_plan=plan,
                accuracy_requirement=requirement,
            ).value
            gradient = gradient.add(plus.subtract(minus).multiply(half))
            shifted_evaluations += 2
        gradients.append(gradient)
    return SplitRealImagDoubleSingleGradientResult(
        expectation=base,
        parameter_order=tuple(sorted(occurrences)),
        gradient=DoubleSingleTensor(
            torch.stack(tuple(value.high for value in gradients)),
            torch.stack(tuple(value.low for value in gradients)),
        ),
        shifted_evaluations=shifted_evaluations,
    )


__all__ = (
    "SplitRealImagDoubleSingleExpectationResult",
    "SplitRealImagDoubleSingleGradientResult",
    "SplitRealImagDoubleSingleStatevectorResult",
    "execute_split_real_imag_double_single_expectation",
    "execute_split_real_imag_double_single_statevector",
    "parameter_shift_split_real_imag_double_single_gradient",
    "split_real_imag_p3_accuracy_envelope",
    "split_real_imag_p3_precision_plan",
)
