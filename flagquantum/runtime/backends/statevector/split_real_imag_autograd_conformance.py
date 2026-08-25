"""CPU reference conformance for the P5 PyTorch autograd bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from .split_real_imag import (
    _complex128_pauli_expectation,
    _normalized_observables,
    _parameter_occurrences,
    _training_conformance_ir,
    _training_conformance_observable,
)
from .split_real_imag_autograd import (
    split_real_imag_device_double_single_autograd_expectation,
)
from .split_real_imag_device_double_single_conformance import _p4_reference_ir
from .split_real_imag_double_single import _bind_p3_ir, _normalized_p3_bindings
from .split_real_imag_double_single_conformance import (
    _complex128_parameter_shift_p3,
)


@dataclass(frozen=True)
class SplitRealImagAutogradConformanceCase:
    depth: int
    seed: int
    forward_absolute_error: float
    gradient_relative_error: float
    gradient_cosine_similarity: float
    finite_difference_gradient_relative_error: float
    passed: bool


@dataclass(frozen=True)
class SplitRealImagAutogradConformanceReport:
    device: str
    provider: str
    cases: tuple[SplitRealImagAutogradConformanceCase, ...]
    passed: bool
    delivered_loss_dtype: str = "float32"
    delivered_tensor_grad_dtype: str = "float32"
    delivered_tensor_grad_precision: str = "float32_boundary"
    internal_gradient_representation: str = "double_single_high_low"
    optimizer_available: bool = True
    optimizer_evidence_in_report: bool = False
    native_cuda_evidence: bool = False
    torch_fl_flagos_evidence: bool = False
    convergence_certification: bool = False
    hardware_certification: bool = False
    schema: str = "flagquantum_split_real_imag_p5_autograd_conformance_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_accepted(self) -> None:
        if not self.passed:
            failed = tuple(
                (case.depth, case.seed) for case in self.cases if not case.passed
            )
            raise RuntimeError(f"split real/imag P5 CPU conformance failed: {failed}")


def _bindings(*, depth: int, seed: int) -> dict[str, torch.Tensor]:
    return {
        "alpha": torch.tensor(
            0.17 + seed * 0.003, dtype=torch.float32, requires_grad=True
        ),
        "beta": torch.tensor(
            -0.29 + depth * 0.0002, dtype=torch.float32, requires_grad=True
        ),
        "gamma": torch.tensor(
            0.11 - seed * 0.002, dtype=torch.float32, requires_grad=True
        ),
    }


def _finite_difference_gradient(
    ir: Any,
    terms: Any,
    bindings: Mapping[str, torch.Tensor],
    parameter_order: Sequence[str],
    *,
    epsilon: float,
) -> torch.Tensor:
    values = []
    for name in parameter_order:
        plus = dict(bindings)
        minus = dict(bindings)
        plus[name] = bindings[name] + epsilon
        minus[name] = bindings[name] - epsilon
        plus_value = _complex128_pauli_expectation(_bind_p3_ir(ir, plus), terms)
        minus_value = _complex128_pauli_expectation(_bind_p3_ir(ir, minus), terms)
        values.append((plus_value - minus_value) / (2.0 * epsilon))
    return torch.stack(values)


def run_split_real_imag_autograd_conformance(
    device: str | torch.device = "cpu",
    *,
    depths: Sequence[int] = (8, 32, 128),
    seeds: Sequence[int] = (0, 7),
    finite_difference_epsilon: float = 1e-3,
    max_forward_absolute_error: float = 1e-6,
    max_gradient_relative_error: float = 1e-5,
    min_gradient_cosine_similarity: float = 0.99999,
    max_finite_difference_gradient_relative_error: float = 2e-5,
) -> SplitRealImagAutogradConformanceReport:
    """Compare the P5 FP32 delivery boundary with CPU complex128 diagnostics."""

    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise NotImplementedError("the first P5 autograd conformance is CPU-only")
    cases: list[SplitRealImagAutogradConformanceCase] = []
    observable = _training_conformance_observable()
    for depth in tuple(int(value) for value in depths):
        for seed in tuple(int(value) for value in seeds):
            ir = _training_conformance_ir(depth, seed)
            bindings = _bindings(depth=depth, seed=seed)
            loss = split_real_imag_device_double_single_autograd_expectation(
                ir,
                observable,
                parameter_bindings=bindings,
                device=resolved,
                preflight=not cases,
            )
            order = tuple(sorted(bindings))
            gradients = torch.autograd.grad(
                loss, tuple(bindings[name] for name in order)
            )
            candidate_gradient = torch.stack(gradients).detach().to(torch.float64)

            reference_ir = _p4_reference_ir(ir)
            reference_bindings = _normalized_p3_bindings(bindings)
            terms = _normalized_observables(reference_ir, observable)
            reference_value, reference_gradient = _complex128_parameter_shift_p3(
                reference_ir,
                terms,
                reference_bindings,
                _parameter_occurrences(reference_ir),
            )
            finite_difference = _finite_difference_gradient(
                reference_ir,
                terms,
                reference_bindings,
                order,
                epsilon=finite_difference_epsilon,
            )
            reference_norm = torch.clamp(
                torch.linalg.vector_norm(reference_gradient), min=1e-15
            )
            gradient_relative_error = float(
                (
                    torch.linalg.vector_norm(candidate_gradient - reference_gradient)
                    / reference_norm
                ).item()
            )
            finite_difference_relative_error = float(
                (
                    torch.linalg.vector_norm(candidate_gradient - finite_difference)
                    / reference_norm
                ).item()
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    candidate_gradient.reshape(1, -1),
                    reference_gradient.reshape(1, -1),
                ).item()
            )
            forward_error = float(
                torch.abs(loss.detach().to(torch.float64) - reference_value).item()
            )
            passed = bool(
                forward_error <= max_forward_absolute_error
                and gradient_relative_error <= max_gradient_relative_error
                and cosine >= min_gradient_cosine_similarity
                and finite_difference_relative_error
                <= max_finite_difference_gradient_relative_error
            )
            cases.append(
                SplitRealImagAutogradConformanceCase(
                    depth=depth,
                    seed=seed,
                    forward_absolute_error=forward_error,
                    gradient_relative_error=gradient_relative_error,
                    gradient_cosine_similarity=cosine,
                    finite_difference_gradient_relative_error=(
                        finite_difference_relative_error
                    ),
                    passed=passed,
                )
            )
    return SplitRealImagAutogradConformanceReport(
        device=str(resolved),
        provider="pytorch_cpu",
        cases=tuple(cases),
        passed=bool(cases) and all(case.passed for case in cases),
    )


__all__ = (
    "SplitRealImagAutogradConformanceCase",
    "SplitRealImagAutogradConformanceReport",
    "run_split_real_imag_autograd_conformance",
)
