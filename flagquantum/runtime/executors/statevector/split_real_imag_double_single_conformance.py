"""Depth-stability conformance for full Double-Single P3 state evolution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from ....core.ir import CircuitIR
from ....core.parameters import Parameter
from .split_real_imag import (
    _complex128_pauli_expectation,
    _normalized_observables,
    _parameter_occurrences,
    _PauliTerm,
    _training_conformance_ir,
    _training_conformance_observable,
)
from .split_real_imag_double_single import (
    SplitRealImagDoubleSingleGradientResult,
    _bind_p3_ir,
    _normalized_p3_bindings,
    parameter_shift_split_real_imag_double_single_gradient,
)
from .split_real_imag_precision import (
    parameter_shift_split_real_imag_precision_gradient,
)


@dataclass(frozen=True)
class SplitRealImagDoubleSingleConformanceCase:
    depth: int
    seed: int
    max_state_abs_error: float
    state_infidelity: float
    norm_drift: float
    expectation_abs_error: float
    p2_expectation_abs_error: float
    expectation_improvement_factor: float
    gradient_relative_error: float
    p2_gradient_relative_error: float
    gradient_improvement_factor: float
    passed: bool


@dataclass(frozen=True)
class SplitRealImagDoubleSingleConformanceReport:
    device: str
    provider: str
    cases: tuple[SplitRealImagDoubleSingleConformanceCase, ...]
    passed: bool
    operator_profile: str
    operator_profile_hash: str
    host_gate_encoding: bool = True
    state_host_fallback: bool = False
    convergence_certification: bool = False
    hardware_certification: bool = False
    schema: str = "flagquantum_split_real_imag_p3_conformance_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_accepted(self) -> None:
        if not self.passed:
            failed = tuple(
                (case.depth, case.seed) for case in self.cases if not case.passed
            )
            raise RuntimeError(f"split real/imag P3 conformance failed: {failed}")


def _improvement(baseline: float, candidate: float) -> float:
    if candidate == 0.0:
        return 1e30 if baseline > 0.0 else 1.0
    return min(baseline / candidate, 1e30)


def _reference_state(ir: CircuitIR) -> torch.Tensor:
    from ....circuit import Circuit

    return Circuit.from_ir(ir, device="cpu", dtype=torch.complex128).state().reshape(-1)


def _complex128_parameter_shift_p3(
    ir: CircuitIR,
    terms: Sequence[_PauliTerm],
    bindings: Mapping[str, torch.Tensor],
    occurrences: Mapping[str, Sequence[tuple[int, str]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    value = _complex128_pauli_expectation(_bind_p3_ir(ir, bindings), terms)
    gradients = []
    for name in sorted(occurrences):
        gradient = torch.zeros((), dtype=torch.float64)
        for instruction_index, parameter_name in occurrences[name]:
            plus = _complex128_pauli_expectation(
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
            )
            minus = _complex128_pauli_expectation(
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
            )
            gradient = gradient + 0.5 * (plus - minus)
        gradients.append(gradient)
    return value, torch.stack(gradients)


def run_split_real_imag_double_single_conformance(
    device: str | torch.device = "cpu",
    *,
    depths: Sequence[int] = (8, 32, 128),
    seeds: Sequence[int] = (0, 7),
    renormalize_every: int = 16,
    max_state_abs_error: float = 1e-10,
    max_state_infidelity: float = 1e-10,
    max_norm_drift: float = 1e-10,
    max_expectation_abs_error: float = 1e-10,
    max_gradient_relative_error: float = 1e-8,
    min_expectation_improvement_factor: float = 100.0,
    min_gradient_improvement_factor: float = 100.0,
) -> SplitRealImagDoubleSingleConformanceReport:
    """Compare full P3 state evolution with P2 and CPU complex128."""

    cases = []
    first_result: SplitRealImagDoubleSingleGradientResult | None = None
    observable = _training_conformance_observable()
    for depth in tuple(int(value) for value in depths):
        for seed in tuple(int(value) for value in seeds):
            ir = _training_conformance_ir(depth, seed)
            bindings: dict[str | Parameter, torch.Tensor] = {
                "alpha": torch.tensor(0.17 + seed * 0.003, dtype=torch.float64),
                "beta": torch.tensor(-0.29 + depth * 0.0002, dtype=torch.float64),
                "gamma": torch.tensor(0.11 - seed * 0.002, dtype=torch.float64),
            }
            result = parameter_shift_split_real_imag_double_single_gradient(
                ir,
                observable,
                parameter_bindings=bindings,
                device=device,
                preflight=first_result is None,
                renormalize_every=renormalize_every,
            )
            p2 = parameter_shift_split_real_imag_precision_gradient(
                ir,
                observable,
                parameter_bindings=bindings,
                device=device,
                preflight=False,
            )
            normalized_bindings = _normalized_p3_bindings(bindings)
            bound_ir = _bind_p3_ir(ir, normalized_bindings)
            terms = _normalized_observables(ir, observable)
            reference_value, reference_gradient = _complex128_parameter_shift_p3(
                ir, terms, normalized_bindings, _parameter_occurrences(ir)
            )
            reference_state = _reference_state(bound_ir)
            candidate_state = result.expectation.state.cpu_complex128()
            state_error = float(
                torch.max(torch.abs(candidate_state - reference_state)).item()
            )
            overlap = torch.sum(torch.conj(reference_state) * candidate_state)
            reference_norm = torch.real(
                torch.sum(torch.conj(reference_state) * reference_state)
            )
            candidate_norm = torch.real(
                torch.sum(torch.conj(candidate_state) * candidate_state)
            )
            infidelity = float(
                torch.abs(
                    1.0
                    - torch.abs(overlap).square() / (reference_norm * candidate_norm)
                ).item()
            )
            norm_drift = float(torch.abs(candidate_norm - 1.0).item())
            candidate_value = result.expectation.cpu_float64()
            candidate_gradient = result.cpu_float64()
            p2_value = p2.expectation.cpu_float64()
            p2_gradient = p2.cpu_float64()
            expectation_error = float(
                torch.abs(candidate_value - reference_value).item()
            )
            p2_expectation_error = float(torch.abs(p2_value - reference_value).item())
            gradient_reference_norm = torch.clamp(
                torch.linalg.vector_norm(reference_gradient), min=1e-15
            )
            gradient_error = float(
                (
                    torch.linalg.vector_norm(candidate_gradient - reference_gradient)
                    / gradient_reference_norm
                ).item()
            )
            p2_gradient_error = float(
                (
                    torch.linalg.vector_norm(p2_gradient - reference_gradient)
                    / gradient_reference_norm
                ).item()
            )
            expectation_improvement = _improvement(
                p2_expectation_error, expectation_error
            )
            gradient_improvement = _improvement(p2_gradient_error, gradient_error)
            passed = bool(
                state_error <= max_state_abs_error
                and infidelity <= max_state_infidelity
                and norm_drift <= max_norm_drift
                and expectation_error <= max_expectation_abs_error
                and gradient_error <= max_gradient_relative_error
                and expectation_improvement >= min_expectation_improvement_factor
                and gradient_improvement >= min_gradient_improvement_factor
            )
            cases.append(
                SplitRealImagDoubleSingleConformanceCase(
                    depth=depth,
                    seed=seed,
                    max_state_abs_error=state_error,
                    state_infidelity=infidelity,
                    norm_drift=norm_drift,
                    expectation_abs_error=expectation_error,
                    p2_expectation_abs_error=p2_expectation_error,
                    expectation_improvement_factor=expectation_improvement,
                    gradient_relative_error=gradient_error,
                    p2_gradient_relative_error=p2_gradient_error,
                    gradient_improvement_factor=gradient_improvement,
                    passed=passed,
                )
            )
            first_result = first_result or result
    assert first_result is not None
    state = first_result.expectation.state
    return SplitRealImagDoubleSingleConformanceReport(
        device=str(state.device),
        provider=state.provider,
        cases=tuple(cases),
        passed=all(case.passed for case in cases),
        operator_profile=state.operator_profile,
        operator_profile_hash=state.operator_profile_hash,
    )


__all__ = (
    "SplitRealImagDoubleSingleConformanceCase",
    "SplitRealImagDoubleSingleConformanceReport",
    "run_split_real_imag_double_single_conformance",
)
