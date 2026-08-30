"""Machine-readable numerical certification for double-single primitives."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .double_single import (
    DoubleSingleComplexTensor,
    DoubleSingleTensor,
    double_single_dot,
    double_single_sum,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

CONFORMANCE_VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DoubleSingleConformanceCase:
    name: str
    reference: float | complex
    candidate: float | complex
    float32_baseline: float | complex
    absolute_error: float
    float32_absolute_error: float
    improvement_factor: float
    max_absolute_error: float
    min_improvement_factor: float
    float32_baseline_within_threshold: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("reference", "candidate", "float32_baseline"):
            value = payload[name]
            if isinstance(value, complex):
                payload[name] = {"real": value.real, "imag": value.imag}
        return payload


@dataclass(frozen=True)
class DoubleSingleConformanceReport:
    schema: str
    conformance_version: str
    torch_version: str
    device: str
    compute_dtype: str
    reference: str
    cases: tuple[DoubleSingleConformanceCase, ...]
    passed: bool
    runtime_integration_certified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "conformance_version": self.conformance_version,
            "torch_version": self.torch_version,
            "device": self.device,
            "compute_dtype": self.compute_dtype,
            "reference": self.reference,
            "passed": self.passed,
            "runtime_integration_certified": self.runtime_integration_certified,
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def require_accepted(self) -> None:
        if not self.passed:
            failed = ", ".join(case.name for case in self.cases if not case.passed)
            raise RuntimeError(f"double-single conformance failed: {failed}")


def _thresholds() -> dict[str, dict[str, float]]:
    contract = tomllib.loads(
        (ROOT / "contracts" / "double-single-contract.toml").read_text(encoding="utf-8")
    )
    return {
        name: {key: float(value) for key, value in raw.items()}
        for name, raw in contract["thresholds"].items()
    }


def _improvement(baseline_error: float, candidate_error: float) -> float:
    if candidate_error == 0.0:
        return 1e30 if baseline_error > 0.0 else 1.0
    return min(baseline_error / candidate_error, 1e30)


def _real_reference_value(value: DoubleSingleTensor) -> torch.Tensor:
    """Reconstruct only after moving FP32 words back to the CPU reference."""

    return value.high.detach().cpu().to(torch.float64) + value.low.detach().cpu().to(
        torch.float64
    )


def _complex_reference_value(value: DoubleSingleComplexTensor) -> torch.Tensor:
    return torch.complex(
        _real_reference_value(value.real), _real_reference_value(value.imag)
    )


def _case(
    name: str,
    reference: float | complex,
    candidate: float | complex,
    baseline: float | complex,
    thresholds: dict[str, dict[str, float]],
) -> DoubleSingleConformanceCase:
    absolute_error = float(abs(candidate - reference))
    baseline_error = float(abs(baseline - reference))
    improvement = _improvement(baseline_error, absolute_error)
    policy = thresholds[name]
    maximum = policy["max_absolute_error"]
    minimum = policy["min_improvement_factor"]
    baseline_within_threshold = baseline_error <= maximum
    passed = (
        math.isfinite(absolute_error)
        and absolute_error <= maximum
        and (baseline_within_threshold or improvement >= minimum)
    )
    return DoubleSingleConformanceCase(
        name,
        reference,
        candidate,
        baseline,
        absolute_error,
        baseline_error,
        improvement,
        maximum,
        minimum,
        baseline_within_threshold,
        passed,
    )


def run_double_single_conformance(
    device: str | torch.device = "cpu",
) -> DoubleSingleConformanceReport:
    """Run bounded cancellation, dot, and complex-chain accuracy probes."""

    resolved = torch.device(device)
    thresholds = _thresholds()
    cases: list[DoubleSingleConformanceCase] = []

    cancellation_reference = torch.tensor([1e8, 1.0, -1e8], dtype=torch.float64).sum()
    cancellation_input = torch.tensor(
        [1e8, 1.0, -1e8], dtype=torch.float32, device=resolved
    )
    cancellation_candidate = _real_reference_value(
        double_single_sum(cancellation_input)
    )
    cancellation_baseline = cancellation_input.sum().cpu().to(torch.float64)
    cases.append(
        _case(
            "cancellation_sum",
            float(cancellation_reference.item()),
            float(cancellation_candidate.item()),
            float(cancellation_baseline.item()),
            thresholds,
        )
    )

    left_reference = torch.tensor([1e8, 1.0, 1e8], dtype=torch.float64)
    right_reference = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    left = left_reference.to(torch.float32).to(resolved)
    right = right_reference.to(torch.float32).to(resolved)
    dot_reference = torch.dot(left_reference, right_reference)
    dot_candidate = _real_reference_value(double_single_dot(left, right))
    dot_baseline = torch.dot(left, right).cpu().to(torch.float64)
    cases.append(
        _case(
            "cancellation_dot",
            float(dot_reference.item()),
            float(dot_candidate.item()),
            float(dot_baseline.item()),
            thresholds,
        )
    )

    angle = 0.00317
    steps = 2048
    factor_reference = torch.polar(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(angle, dtype=torch.float64),
    ).to(torch.complex128)
    reference_value = factor_reference**steps
    pair = DoubleSingleComplexTensor.from_complex128(
        torch.tensor(1.0 + 0.0j, dtype=torch.complex128)
    ).to(resolved)
    factor = DoubleSingleComplexTensor.from_complex128(factor_reference).to(resolved)
    baseline = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=resolved)
    factor32 = factor_reference.to(torch.complex64).to(resolved)
    for _ in range(steps):
        pair = pair.multiply(factor)
        baseline = baseline * factor32
    candidate_value = complex(_complex_reference_value(pair).item())
    baseline_value = complex(baseline.cpu().to(torch.complex128).item())
    cases.append(
        _case(
            "complex_phase_chain",
            complex(reference_value.item()),
            candidate_value,
            baseline_value,
            thresholds,
        )
    )

    results = tuple(cases)
    return DoubleSingleConformanceReport(
        schema="flagquantum_double_single_conformance_v1",
        conformance_version=CONFORMANCE_VERSION,
        torch_version=str(torch.__version__),
        device=str(resolved),
        compute_dtype="float32",
        reference="cpu_float64_and_complex128",
        cases=results,
        passed=all(case.passed for case in results),
    )


__all__ = (
    "CONFORMANCE_VERSION",
    "DoubleSingleConformanceCase",
    "DoubleSingleConformanceReport",
    "run_double_single_conformance",
)
