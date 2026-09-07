"""Conformance for P4 device-generated Double-Single gates and state evolution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import torch

from ....core.parameters import Parameter, ParameterExpression
from ....numerics.double_single import DoubleSingleTensor, double_single_sin_cos
from ....providers.platform import resolve_platform_device
from ....simulation.double_single_device_gates import encode_device_double_single_matrix
from ....simulation.statevector.double_single_host_gates import host_matrix_complex128
from .split_real_imag import (
    _normalized_observables,
    _parameter_occurrences,
    _training_conformance_ir,
    _training_conformance_observable,
)
from .split_real_imag_device_double_single import (
    SplitRealImagDeviceDoubleSingleGradientResult,
    execute_split_real_imag_device_double_single_expectation,
    parameter_shift_split_real_imag_device_double_single_gradient,
)
from .split_real_imag_double_single import (
    _bind_p3_ir,
    _normalized_p3_bindings,
)
from .split_real_imag_double_single_conformance import (
    _complex128_parameter_shift_p3,
    _reference_state,
)


@dataclass(frozen=True)
class SplitRealImagDeviceDoubleSingleConformanceCase:
    depth: int
    seed: int
    max_state_abs_error: float
    state_infidelity: float
    norm_drift: float
    expectation_abs_error: float
    gradient_relative_error: float
    passed: bool


@dataclass(frozen=True)
class SplitRealImagDeviceDoubleSingleStabilityCase:
    depth: int
    seed: int
    max_state_abs_error: float
    state_infidelity: float
    norm_drift: float
    passed: bool


@dataclass(frozen=True)
class SplitRealImagDeviceDoubleSingleConformanceReport:
    device: str
    provider: str
    cases: tuple[SplitRealImagDeviceDoubleSingleConformanceCase, ...]
    stability_cases: tuple[SplitRealImagDeviceDoubleSingleStabilityCase, ...]
    max_trigonometry_abs_error: float
    max_gate_entry_abs_error: float
    passed: bool
    operator_profile: str
    operator_profile_hash: str
    device_only_double_single_trigonometry: bool = True
    host_gate_encoding: bool = False
    parameter_host_fallback: bool = False
    state_host_fallback: bool = False
    convergence_certification: bool = False
    hardware_certification: bool = False
    schema: str = "flagquantum_split_real_imag_p4_conformance_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_accepted(self) -> None:
        if not self.passed:
            raise RuntimeError("split real/imag P4 conformance failed")


def _bindings(
    *, depth: int, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "alpha": torch.tensor(0.17 + seed * 0.003, dtype=torch.float32, device=device),
        "beta": torch.tensor(
            -0.29 + depth * 0.0002, dtype=torch.float32, device=device
        ),
        "gamma": torch.tensor(0.11 - seed * 0.002, dtype=torch.float32, device=device),
    }


def _state_errors(
    candidate: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float, float]:
    state_error = float(torch.max(torch.abs(candidate - reference)).item())
    overlap = torch.sum(torch.conj(reference) * candidate)
    reference_norm = torch.real(torch.sum(torch.conj(reference) * reference))
    candidate_norm = torch.real(torch.sum(torch.conj(candidate) * candidate))
    infidelity = float(
        torch.abs(
            1.0 - torch.abs(overlap).square() / (reference_norm * candidate_norm)
        ).item()
    )
    norm_drift = float(torch.abs(candidate_norm - 1.0).item())
    return state_error, infidelity, norm_drift


def _p4_reference_ir(ir: Any) -> Any:
    instructions = []
    for instruction in ir.instructions:
        params = {
            name: (
                value
                if isinstance(value, (Parameter, ParameterExpression))
                else torch.as_tensor(value, dtype=torch.float32)
                .to(torch.float64)
                .reshape(())
            )
            for name, value in instruction.params.items()
        }
        instructions.append(replace(instruction, params=params))
    return replace(ir, instructions=tuple(instructions))


def _trigonometry_error(device: torch.device) -> float:
    reference = torch.tensor(
        (
            -1024.0,
            -100.0,
            -math.pi,
            -math.pi / 2.0,
            -0.7,
            -1e-8,
            0.0,
            1e-8,
            0.7,
            math.pi / 2.0,
            math.pi,
            100.0,
            1024.0,
        ),
        dtype=torch.float64,
    )
    value = DoubleSingleTensor.from_float64(reference).to(device)
    sine, cosine = double_single_sin_cos(value)
    reconstructed_sine = sine.high.detach().cpu().to(
        torch.float64
    ) + sine.low.detach().cpu().to(torch.float64)
    reconstructed_cosine = cosine.high.detach().cpu().to(
        torch.float64
    ) + cosine.low.detach().cpu().to(torch.float64)
    return float(
        torch.max(
            torch.stack(
                (
                    torch.max(torch.abs(reconstructed_sine - torch.sin(reference))),
                    torch.max(torch.abs(reconstructed_cosine - torch.cos(reference))),
                )
            )
        ).item()
    )


def _gate_error(device: torch.device) -> float:
    errors = []
    for name in ("rx", "ry", "rz", "rxx", "ryy", "rzz"):
        wires = (0, 1) if name in {"rxx", "ryy", "rzz"} else (0,)
        for angle in (-100.0, -math.pi, -0.7, 0.0, 0.7, math.pi, 100.0):
            reference_angle = torch.tensor(angle, dtype=torch.float64)
            pair = DoubleSingleTensor.from_float64(reference_angle).to(device)
            from ....core.ir import Instruction

            instruction = Instruction(name=name, wires=wires, params={"theta": pair})
            matrix, _ = encode_device_double_single_matrix(
                instruction, bindings={}, device=device
            )
            reconstructed = matrix.to("cpu").to_complex128()
            reference_instruction = Instruction(
                name=name, wires=wires, params={"theta": reference_angle}
            )
            errors.append(
                torch.max(
                    torch.abs(
                        reconstructed - host_matrix_complex128(reference_instruction)
                    )
                )
            )
    return float(torch.max(torch.stack(errors)).item())


def run_split_real_imag_device_double_single_conformance(
    device: str | torch.device = "cpu",
    *,
    depths: Sequence[int] = (8, 32, 128),
    seeds: Sequence[int] = (0, 7),
    stability_depths: Sequence[int] = (512,),
    stability_seeds: Sequence[int] = (0,),
    renormalize_every: int = 16,
    max_trigonometry_abs_error: float = 1e-11,
    max_gate_entry_abs_error: float = 1e-11,
    max_state_abs_error: float = 1e-9,
    max_state_infidelity: float = 1e-9,
    max_norm_drift: float = 1e-9,
    max_expectation_abs_error: float = 1e-9,
    max_gradient_relative_error: float = 1e-8,
) -> SplitRealImagDeviceDoubleSingleConformanceReport:
    resolved_device = resolve_platform_device(device)
    observable = _training_conformance_observable()
    cases = []
    first_result: SplitRealImagDeviceDoubleSingleGradientResult | None = None
    for depth in tuple(int(value) for value in depths):
        for seed in tuple(int(value) for value in seeds):
            ir = _training_conformance_ir(depth, seed)
            bindings = _bindings(depth=depth, seed=seed, device=resolved_device)
            result = parameter_shift_split_real_imag_device_double_single_gradient(
                ir,
                observable,
                parameter_bindings=bindings,
                device=resolved_device,
                preflight=first_result is None,
                renormalize_every=renormalize_every,
            )
            reference_ir = _p4_reference_ir(ir)
            reference_bindings = _normalized_p3_bindings(bindings)
            terms = _normalized_observables(reference_ir, observable)
            reference_value, reference_gradient = _complex128_parameter_shift_p3(
                reference_ir,
                terms,
                reference_bindings,
                _parameter_occurrences(reference_ir),
            )
            reference_state = _reference_state(
                _bind_p3_ir(reference_ir, reference_bindings)
            )
            state_error, infidelity, norm_drift = _state_errors(
                result.expectation.state.cpu_complex128(), reference_state
            )
            expectation_error = float(
                torch.abs(result.expectation.cpu_float64() - reference_value).item()
            )
            gradient_norm = torch.clamp(
                torch.linalg.vector_norm(reference_gradient), min=1e-15
            )
            gradient_error = float(
                (
                    torch.linalg.vector_norm(result.cpu_float64() - reference_gradient)
                    / gradient_norm
                ).item()
            )
            passed = bool(
                state_error <= max_state_abs_error
                and infidelity <= max_state_infidelity
                and norm_drift <= max_norm_drift
                and expectation_error <= max_expectation_abs_error
                and gradient_error <= max_gradient_relative_error
            )
            cases.append(
                SplitRealImagDeviceDoubleSingleConformanceCase(
                    depth=depth,
                    seed=seed,
                    max_state_abs_error=state_error,
                    state_infidelity=infidelity,
                    norm_drift=norm_drift,
                    expectation_abs_error=expectation_error,
                    gradient_relative_error=gradient_error,
                    passed=passed,
                )
            )
            first_result = first_result or result

    stability_cases = []
    for depth in tuple(int(value) for value in stability_depths):
        for seed in tuple(int(value) for value in stability_seeds):
            ir = _training_conformance_ir(depth, seed)
            bindings = _bindings(depth=depth, seed=seed, device=resolved_device)
            stability_result = execute_split_real_imag_device_double_single_expectation(
                ir,
                observable,
                parameter_bindings=bindings,
                device=resolved_device,
                preflight=False,
                renormalize_every=renormalize_every,
            )
            reference_ir = _p4_reference_ir(ir)
            reference_bindings = _normalized_p3_bindings(bindings)
            reference_state = _reference_state(
                _bind_p3_ir(reference_ir, reference_bindings)
            )
            state_error, infidelity, norm_drift = _state_errors(
                stability_result.state.cpu_complex128(), reference_state
            )
            passed = bool(
                state_error <= max_state_abs_error
                and infidelity <= max_state_infidelity
                and norm_drift <= max_norm_drift
            )
            stability_cases.append(
                SplitRealImagDeviceDoubleSingleStabilityCase(
                    depth=depth,
                    seed=seed,
                    max_state_abs_error=state_error,
                    state_infidelity=infidelity,
                    norm_drift=norm_drift,
                    passed=passed,
                )
            )
    assert first_result is not None
    trig_error = _trigonometry_error(resolved_device)
    gate_error = _gate_error(resolved_device)
    state = first_result.expectation.state
    return SplitRealImagDeviceDoubleSingleConformanceReport(
        device=str(state.device),
        provider=state.provider,
        cases=tuple(cases),
        stability_cases=tuple(stability_cases),
        max_trigonometry_abs_error=trig_error,
        max_gate_entry_abs_error=gate_error,
        passed=bool(
            all(case.passed for case in cases)
            and all(case.passed for case in stability_cases)
            and trig_error <= max_trigonometry_abs_error
            and gate_error <= max_gate_entry_abs_error
        ),
        operator_profile=state.operator_profile,
        operator_profile_hash=state.operator_profile_hash,
    )


__all__ = (
    "SplitRealImagDeviceDoubleSingleConformanceCase",
    "SplitRealImagDeviceDoubleSingleConformanceReport",
    "SplitRealImagDeviceDoubleSingleStabilityCase",
    "run_split_real_imag_device_double_single_conformance",
)
