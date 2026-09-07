"""Selective Double-Single reductions for the split FP32 statevector."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, cast

import torch

from ....core.ir import CircuitIR, ensure_circuit_ir
from ....core.numerics import (
    AccuracyMode,
    AccuracyRequirementContract,
    ComplexRepresentation,
    PrecisionPlanContract,
    RefinementStrategy,
)
from ....core.parameters import Parameter
from ....numerics.double_single import (
    DoubleSingleTensor,
    double_single_sum,
)
from ....simulation.statevector.split_real_imag import (
    double_single_pauli_term_expectation,
)
from .split_real_imag import (
    SplitRealImagStatevectorResult,
    _bind_p1_ir,
    _coerce_bounded_accuracy,
    _coerce_exact_precision_plan,
    _complex128_parameter_shift,
    _execute_bound_split_statevector,
    _normalized_bindings,
    _normalized_observables,
    _parameter_occurrences,
    _PauliTerm,
    _training_conformance_ir,
    _validate_p1_execution_scope,
    parameter_shift_split_real_imag_gradient,
)

P2_EXECUTOR = "split_real_imag_statevector_p2_precision"
P2_PROFILE = "split_real_imag_statevector_p2_precision"


def split_real_imag_p2_precision_plan() -> PrecisionPlanContract:
    """Return the only precision plan implemented by the P2 executor."""

    return PrecisionPlanContract(
        complex_representation=ComplexRepresentation.SPLIT_REAL_IMAG.value,
        parameter_dtype="float32",
        gate_generation_dtype="float32",
        state_storage_dtype="float32",
        kernel_compute_dtype="float32",
        reduction_dtype="double_single_fp32",
        decomposition_dtype="not_applicable",
        gradient_dtype="double_single_fp32",
        optimizer_master_dtype="not_integrated",
        communication_dtype="not_applicable",
        checkpoint_dtype="float32",
        refinement=RefinementStrategy.COMPENSATED_REDUCTION.value,
        allow_dtype_demotion=False,
    )


def split_real_imag_p2_accuracy_envelope() -> AccuracyRequirementContract:
    """Return the bounded numerical envelope certified by P2 conformance."""

    return AccuracyRequirementContract(
        mode=AccuracyMode.ADAPTIVE.value,
        max_norm_drift=5e-5,
        max_expectation_abs_error=5e-5,
        max_expectation_rel_error=5e-4,
        max_gradient_rel_error=2e-3,
        min_gradient_cosine_similarity=0.999,
        require_determinism=False,
        require_convergence_evidence=False,
    )


def _coerce_p2_precision_plan(
    value: PrecisionPlanContract | Mapping[str, Any] | None,
) -> PrecisionPlanContract:
    return _coerce_exact_precision_plan(
        value,
        implemented=split_real_imag_p2_precision_plan(),
        error=(
            "split real/imag P2 implements selective Double-Single reductions "
            "only; the requested precision plan is not executable"
        ),
    )


def _coerce_p2_accuracy_requirement(
    value: AccuracyRequirementContract | Mapping[str, Any] | None,
) -> AccuracyRequirementContract:
    certified = split_real_imag_p2_accuracy_envelope()
    requested = _coerce_bounded_accuracy(
        value,
        certified=certified,
        maximum_fields=(
            "max_norm_drift",
            "max_expectation_abs_error",
            "max_expectation_rel_error",
            "max_gradient_rel_error",
        ),
        owner="split real/imag P2",
    )
    unsupported = {
        name: getattr(requested, name)
        for name in (
            "max_state_infidelity",
            "max_decomposition_residual",
            "max_truncation_error",
        )
        if getattr(requested, name) is not None
    }
    if requested.require_determinism:
        unsupported["require_determinism"] = True
    if requested.require_convergence_evidence:
        unsupported["require_convergence_evidence"] = True
    if unsupported:
        raise RuntimeError(
            "split real/imag P2 lacks evidence for requested constraints: "
            f"{unsupported}"
        )
    return requested


@dataclass(frozen=True)
class SplitRealImagPrecisionExpectationResult:
    """Pauli expectation retained as Double-Single high/low FP32 words."""

    state: SplitRealImagStatevectorResult
    value: DoubleSingleTensor
    term_expectations: DoubleSingleTensor
    term_count: int
    precision_plan: PrecisionPlanContract
    accuracy_requirement: AccuracyRequirementContract

    def cpu_float64(self) -> torch.Tensor:
        """Reconstruct the scalar after moving both FP32 words to CPU."""

        return self.value.to("cpu").to_float64().detach()

    def summary(self) -> dict[str, Any]:
        summary = cast(dict[str, Any], self.state.summary())
        summary.update(
            {
                "schema": "flagquantum_split_real_imag_precision_expectation_v1",
                "term_count": self.term_count,
                "value_high_dtype": "float32",
                "value_low_dtype": "float32",
                "precision_plan": self.precision_plan.to_dict(),
                "precision_plan_hash": self.precision_plan.content_hash(),
                "accuracy_requirement": self.accuracy_requirement.to_dict(),
                "accuracy_requirement_hash": self.accuracy_requirement.content_hash(),
                "convergence_evidence": False,
            }
        )
        return summary


@dataclass(frozen=True)
class SplitRealImagPrecisionGradientResult:
    """Parameter-shift gradient retained as Double-Single FP32 words."""

    expectation: SplitRealImagPrecisionExpectationResult
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
                "schema": "flagquantum_split_real_imag_precision_gradient_v1",
                "gradient_method": "parameter_shift",
                "parameter_order": self.parameter_order,
                "shifted_evaluations": self.shifted_evaluations,
                "gradient_high_dtype": "float32",
                "gradient_low_dtype": "float32",
            }
        )
        return summary


def _expectation_from_bound_p2_ir(
    ir: CircuitIR,
    terms: Sequence[_PauliTerm],
    *,
    device: str | torch.device,
    preflight: bool,
    precision_plan: PrecisionPlanContract,
    accuracy_requirement: AccuracyRequirementContract,
) -> SplitRealImagPrecisionExpectationResult:
    state = _execute_bound_split_statevector(
        ir,
        device=device,
        profile_name=P2_PROFILE,
        preflight=preflight,
        executor=P2_EXECUTOR,
    )
    values = tuple(
        double_single_pauli_term_expectation(
            state.real,
            state.imag,
            term.ops,
            term.coefficient,
            n_wires=ir.n_wires,
        )
        for term in terms
    )
    term_values = DoubleSingleTensor(
        torch.stack(tuple(value.high for value in values)),
        torch.stack(tuple(value.low for value in values)),
    )
    return SplitRealImagPrecisionExpectationResult(
        state=state,
        value=double_single_sum(term_values),
        term_expectations=term_values,
        term_count=len(terms),
        precision_plan=precision_plan,
        accuracy_requirement=accuracy_requirement,
    )


def execute_split_real_imag_precision_expectation(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, Any] | None = None,
    device: str | torch.device = "cpu",
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None = None,
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None = None,
    preflight: bool = True,
) -> SplitRealImagPrecisionExpectationResult:
    """Evaluate an explicit P2 selective-Double-Single Pauli expectation."""

    plan = _coerce_p2_precision_plan(precision_plan)
    requirement = _coerce_p2_accuracy_requirement(accuracy_requirement)
    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    occurrences = _parameter_occurrences(ir)
    bindings = _normalized_bindings(parameter_bindings or {})
    if set(bindings) != set(occurrences):
        raise ValueError(
            "parameter bindings must exactly match named circuit parameters; "
            f"expected {sorted(occurrences)}, got {sorted(bindings)}"
        )
    terms = _normalized_observables(ir, observable)
    return _expectation_from_bound_p2_ir(
        _bind_p1_ir(ir, bindings),
        terms,
        device=device,
        preflight=preflight,
        precision_plan=plan,
        accuracy_requirement=requirement,
    )


def parameter_shift_split_real_imag_precision_gradient(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, Any],
    device: str | torch.device = "cpu",
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None = None,
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None = None,
    preflight: bool = True,
) -> SplitRealImagPrecisionGradientResult:
    """Return P2 parameter-shift gradients with high/low FP32 words."""

    plan = _coerce_p2_precision_plan(precision_plan)
    requirement = _coerce_p2_accuracy_requirement(accuracy_requirement)
    ir = ensure_circuit_ir(circuit_or_ir)
    _validate_p1_execution_scope(circuit_or_ir, ir)
    occurrences = _parameter_occurrences(ir)
    bindings = _normalized_bindings(parameter_bindings)
    if set(bindings) != set(occurrences) or not occurrences:
        raise ValueError(
            "parameter bindings must exactly match at least one named circuit "
            f"parameter; expected {sorted(occurrences)}, got {sorted(bindings)}"
        )
    terms = _normalized_observables(ir, observable)
    base = _expectation_from_bound_p2_ir(
        _bind_p1_ir(ir, bindings),
        terms,
        device=device,
        preflight=preflight,
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
            plus = _expectation_from_bound_p2_ir(
                _bind_p1_ir(
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
                precision_plan=plan,
                accuracy_requirement=requirement,
            ).value
            minus = _expectation_from_bound_p2_ir(
                _bind_p1_ir(
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
                precision_plan=plan,
                accuracy_requirement=requirement,
            ).value
            gradient = gradient.add(plus.subtract(minus).multiply(half))
            shifted_evaluations += 2
        gradients.append(gradient)
    return SplitRealImagPrecisionGradientResult(
        expectation=base,
        parameter_order=tuple(sorted(occurrences)),
        gradient=DoubleSingleTensor(
            torch.stack(tuple(value.high for value in gradients)),
            torch.stack(tuple(value.low for value in gradients)),
        ),
        shifted_evaluations=shifted_evaluations,
    )


@dataclass(frozen=True)
class SplitRealImagPrecisionConformanceCase:
    depth: int
    seed: int
    expectation_abs_error: float
    fp32_expectation_abs_error: float
    expectation_improvement_factor: float
    gradient_relative_error: float
    fp32_gradient_relative_error: float
    gradient_improvement_factor: float
    passed: bool


@dataclass(frozen=True)
class SplitRealImagPrecisionConformanceReport:
    device: str
    provider: str
    cases: tuple[SplitRealImagPrecisionConformanceCase, ...]
    passed: bool
    operator_profile: str
    operator_profile_hash: str
    hardware_certification: bool = False
    schema: str = "flagquantum_split_real_imag_precision_conformance_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_accepted(self) -> None:
        if not self.passed:
            failed = tuple(
                (case.depth, case.seed) for case in self.cases if not case.passed
            )
            raise RuntimeError(f"split real/imag P2 conformance failed: {failed}")


def _improvement(baseline: float, candidate: float) -> float:
    if candidate == 0.0:
        return 1e30 if baseline > 0.0 else 1.0
    return min(baseline / candidate, 1e30)


def _cancellation_terms() -> tuple[_PauliTerm, ...]:
    # A single small term is reduction-order sensitive: CUDA may pair the two
    # large terms first and accidentally make the FP32 baseline exact. Eight
    # small terms keep the workload cancellation-sensitive on both CPU and GPU.
    return tuple(
        _PauliTerm(torch.tensor(coefficient, dtype=torch.float32), ((0, "z"),))
        for coefficient in (1e8, *(1.0 for _ in range(8)), -1e8)
    )


def run_split_real_imag_precision_conformance(
    device: str | torch.device = "cpu",
    *,
    depths: Sequence[int] = (8, 32, 128),
    seeds: Sequence[int] = (0, 7),
    max_expectation_abs_error: float = 5e-5,
    max_gradient_relative_error: float = 2e-3,
    min_expectation_improvement_factor: float = 100.0,
    min_gradient_improvement_factor: float = 100.0,
) -> SplitRealImagPrecisionConformanceReport:
    """Compare selective Double-Single reductions with FP32 and complex128."""

    cases = []
    first_result = None
    terms = _cancellation_terms()
    for depth in tuple(int(value) for value in depths):
        for seed in tuple(int(value) for value in seeds):
            ir = _training_conformance_ir(depth, seed)
            bindings = {
                "alpha": torch.tensor(0.17 + seed * 0.003),
                "beta": torch.tensor(-0.29 + depth * 0.0002),
                "gamma": torch.tensor(0.11 - seed * 0.002),
            }
            result = parameter_shift_split_real_imag_precision_gradient(
                ir,
                terms,
                parameter_bindings=bindings,
                device=device,
                preflight=first_result is None,
            )
            baseline = parameter_shift_split_real_imag_gradient(
                ir,
                terms,
                parameter_bindings=bindings,
                device=device,
                preflight=False,
            )
            reference_value, reference_gradient = _complex128_parameter_shift(
                ir,
                terms,
                _normalized_bindings(bindings),
                _parameter_occurrences(ir),
            )
            candidate_value = result.expectation.cpu_float64()
            candidate_gradient = result.cpu_float64()
            baseline_value = baseline.expectation.value.detach().cpu().to(torch.float64)
            baseline_gradient = baseline.gradient.detach().cpu().to(torch.float64)
            expectation_error = float(
                torch.abs(candidate_value - reference_value).item()
            )
            baseline_expectation_error = float(
                torch.abs(baseline_value - reference_value).item()
            )
            reference_norm = torch.clamp(
                torch.linalg.vector_norm(reference_gradient), min=1e-12
            )
            gradient_error = float(
                (
                    torch.linalg.vector_norm(candidate_gradient - reference_gradient)
                    / reference_norm
                ).item()
            )
            baseline_gradient_error = float(
                (
                    torch.linalg.vector_norm(baseline_gradient - reference_gradient)
                    / reference_norm
                ).item()
            )
            expectation_improvement = _improvement(
                baseline_expectation_error, expectation_error
            )
            gradient_improvement = _improvement(baseline_gradient_error, gradient_error)
            passed = bool(
                expectation_error <= max_expectation_abs_error
                and gradient_error <= max_gradient_relative_error
                and expectation_improvement >= min_expectation_improvement_factor
                and gradient_improvement >= min_gradient_improvement_factor
            )
            cases.append(
                SplitRealImagPrecisionConformanceCase(
                    depth=depth,
                    seed=seed,
                    expectation_abs_error=expectation_error,
                    fp32_expectation_abs_error=baseline_expectation_error,
                    expectation_improvement_factor=expectation_improvement,
                    gradient_relative_error=gradient_error,
                    fp32_gradient_relative_error=baseline_gradient_error,
                    gradient_improvement_factor=gradient_improvement,
                    passed=passed,
                )
            )
            first_result = first_result or result
    assert first_result is not None
    return SplitRealImagPrecisionConformanceReport(
        device=str(first_result.expectation.state.device),
        provider=first_result.expectation.state.provider,
        cases=tuple(cases),
        passed=all(case.passed for case in cases),
        operator_profile=first_result.expectation.state.operator_profile,
        operator_profile_hash=first_result.expectation.state.operator_profile_hash,
    )


__all__ = (
    "SplitRealImagPrecisionConformanceCase",
    "SplitRealImagPrecisionConformanceReport",
    "SplitRealImagPrecisionExpectationResult",
    "SplitRealImagPrecisionGradientResult",
    "execute_split_real_imag_precision_expectation",
    "parameter_shift_split_real_imag_precision_gradient",
    "run_split_real_imag_precision_conformance",
    "split_real_imag_p2_accuracy_envelope",
    "split_real_imag_p2_precision_plan",
)
