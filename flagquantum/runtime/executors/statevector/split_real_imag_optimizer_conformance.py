"""Single-device trajectory evidence for the P5 Double-Single SGD optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from ....algorithms import pauli_term
from ....circuit import Circuit
from ....core.ir import ensure_circuit_ir
from ....core.parameters import Parameter
from ....providers.platform import get_platform_runtime, resolve_platform_device
from ....simulation.numerics.double_single import DoubleSingleTensor
from .split_real_imag import _normalized_observables, _parameter_occurrences
from .split_real_imag_autograd_optimizer import (
    double_single_sgd_step,
    initialize_split_real_imag_double_single_sgd,
    split_real_imag_double_single_sgd_step,
)
from .split_real_imag_device_double_single import (
    execute_split_real_imag_device_double_single_expectation,
    parameter_shift_split_real_imag_device_double_single_gradient,
)
from .split_real_imag_device_double_single_conformance import _p4_reference_ir
from .split_real_imag_double_single_conformance import (
    _complex128_parameter_shift_p3,
)


@dataclass(frozen=True)
class SplitRealImagOptimizerTrajectoryCase:
    workload: str
    step: int
    double_single_parameter_absolute_error: float
    float32_parameter_absolute_error: float
    double_single_loss_absolute_error: float
    float32_loss_absolute_error: float
    passed: bool


@dataclass(frozen=True)
class SplitRealImagOptimizerConformanceReport:
    device: str
    provider: str
    cases: tuple[SplitRealImagOptimizerTrajectoryCase, ...]
    cancellation_double_single_absolute_error: float
    cancellation_float32_absolute_error: float
    passed: bool
    optimizer: str = "double_single_sgd_without_momentum"
    parameter_representation: str = "double_single_high_low"
    gradient_representation: str = "double_single_high_low"
    tensor_grad_used: bool = False
    native_cuda_evidence: bool = False
    torch_fl_flagos_evidence: bool = False
    convergence_certification: bool = False
    schema: str = "flagquantum_split_real_imag_p5_optimizer_conformance_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_accepted(self) -> None:
        if not self.passed:
            failed = tuple(
                (case.workload, case.step) for case in self.cases if not case.passed
            )
            raise RuntimeError(
                f"split real/imag P5 optimizer conformance failed: {failed}"
            )


def _vqe_workload() -> tuple[Any, Any, dict[str, float], float]:
    alpha = Parameter("alpha")
    beta = Parameter("beta")
    circuit = Circuit(2).ry(0, theta=alpha).ry(1, theta=beta).cx(0, 1)
    observable = (
        pauli_term(0.7, "Z", (0,)),
        pauli_term(-0.4, "Z", (1,)),
        pauli_term(0.2, "XX", (0, 1)),
    )
    return circuit, observable, {"alpha": 0.31, "beta": -0.27}, 0.05


def _qaoa_workload() -> tuple[Any, Any, dict[str, float], float]:
    gamma = Parameter("gamma")
    beta = Parameter("beta")
    circuit = Circuit(3)
    for wire in range(3):
        circuit.h(wire)
    circuit.rzz(0, 1, theta=gamma).rzz(1, 2, theta=gamma)
    for wire in range(3):
        circuit.rx(wire, theta=beta)
    observable = (
        pauli_term(1.0, "ZZ", (0, 1)),
        pauli_term(0.75, "ZZ", (1, 2)),
        pauli_term(0.1, "Z", (1,)),
    )
    return circuit, observable, {"beta": 0.19, "gamma": -0.23}, 0.04


def _reference_value_gradient(
    reference_ir: Any,
    terms: Any,
    bindings: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return _complex128_parameter_shift_p3(
        reference_ir,
        terms,
        bindings,
        _parameter_occurrences(reference_ir),
    )


def _candidate_loss(
    ir: Any,
    observable: Any,
    bindings: Mapping[str, Any],
    *,
    device: torch.device,
) -> float:
    return float(
        execute_split_real_imag_device_double_single_expectation(
            ir,
            observable,
            parameter_bindings=bindings,
            device=device,
            preflight=False,
        )
        .cpu_float64()
        .item()
    )


def _cancellation_case(steps: int, *, device: torch.device) -> tuple[float, float]:
    state = initialize_split_real_imag_double_single_sgd(
        {"theta": torch.tensor(1.0, dtype=torch.float32, device=device)}
    )
    gradient = DoubleSingleTensor.from_float32(
        torch.tensor([2.0**-31], dtype=torch.float32, device=device)
    )
    float32_parameter = torch.tensor([1.0], dtype=torch.float32, device=device)
    learning_rate = torch.tensor(1.0, dtype=torch.float32, device=device)
    for _ in range(steps):
        state, _ = double_single_sgd_step(
            state,
            gradient,
            parameter_order=("theta",),
            learning_rate=learning_rate,
        )
        float32_parameter -= gradient.to_float32()
    reference = torch.tensor([1.0 - steps * 2.0**-31], dtype=torch.float64)
    double_single_error = float(
        torch.max(torch.abs(state.cpu_float64() - reference)).item()
    )
    float32_error = float(
        torch.max(
            torch.abs(float32_parameter.detach().cpu().to(torch.float64) - reference)
        ).item()
    )
    return double_single_error, float32_error


def _trajectory(
    workload: str,
    circuit: Any,
    observable: Any,
    initial: Mapping[str, float],
    learning_rate: float,
    *,
    checkpoints: Sequence[int],
    max_parameter_absolute_error: float,
    max_loss_absolute_error: float,
    preflight: bool,
    device: torch.device,
) -> tuple[SplitRealImagOptimizerTrajectoryCase, ...]:
    order = tuple(sorted(initial))
    initial_float32 = {
        name: torch.tensor(initial[name], dtype=torch.float32, device=device)
        for name in order
    }
    state = initialize_split_real_imag_double_single_sgd(initial_float32)
    float32_parameters = torch.stack(tuple(initial_float32[name] for name in order))
    reference_parameters = float32_parameters.detach().cpu().to(torch.float64)
    ir = ensure_circuit_ir(circuit)
    reference_ir = _p4_reference_ir(ir)
    terms = _normalized_observables(reference_ir, observable)
    device_learning_rate = torch.tensor(
        learning_rate, dtype=torch.float32, device=device
    )
    cases: list[SplitRealImagOptimizerTrajectoryCase] = []
    for step in range(1, max(checkpoints) + 1):
        candidate = split_real_imag_double_single_sgd_step(
            ir,
            observable,
            state,
            learning_rate=device_learning_rate,
            preflight=preflight and step == 1,
        )
        state = candidate.state

        float32_bindings = {
            name: float32_parameters[index] for index, name in enumerate(order)
        }
        float32_gradient = (
            parameter_shift_split_real_imag_device_double_single_gradient(
                ir,
                observable,
                parameter_bindings=float32_bindings,
                device=device,
                preflight=False,
            ).gradient.to_float32()
        )
        float32_parameters = (
            float32_parameters - device_learning_rate * float32_gradient
        )

        reference_bindings = {
            name: reference_parameters[index] for index, name in enumerate(order)
        }
        _, reference_gradient = _reference_value_gradient(
            reference_ir, terms, reference_bindings
        )
        reference_parameters = reference_parameters - learning_rate * reference_gradient

        if step not in checkpoints:
            continue
        next_reference_bindings = {
            name: reference_parameters[index] for index, name in enumerate(order)
        }
        reference_loss, _ = _reference_value_gradient(
            reference_ir, terms, next_reference_bindings
        )
        double_single_loss = _candidate_loss(
            ir, observable, state.bindings(), device=device
        )
        next_float32_bindings = {
            name: float32_parameters[index] for index, name in enumerate(order)
        }
        float32_loss = _candidate_loss(
            ir, observable, next_float32_bindings, device=device
        )
        double_single_parameter_error = float(
            torch.max(torch.abs(state.cpu_float64() - reference_parameters)).item()
        )
        float32_parameter_error = float(
            torch.max(
                torch.abs(
                    float32_parameters.detach().cpu().to(torch.float64)
                    - reference_parameters
                )
            ).item()
        )
        double_single_loss_error = abs(
            double_single_loss - float(reference_loss.item())
        )
        float32_loss_error = abs(float32_loss - float(reference_loss.item()))
        cases.append(
            SplitRealImagOptimizerTrajectoryCase(
                workload=workload,
                step=step,
                double_single_parameter_absolute_error=double_single_parameter_error,
                float32_parameter_absolute_error=float32_parameter_error,
                double_single_loss_absolute_error=double_single_loss_error,
                float32_loss_absolute_error=float32_loss_error,
                passed=bool(
                    double_single_parameter_error <= max_parameter_absolute_error
                    and double_single_loss_error <= max_loss_absolute_error
                    and double_single_parameter_error <= float32_parameter_error
                    and double_single_loss_error <= float32_loss_error
                ),
            )
        )
    return tuple(cases)


def run_split_real_imag_optimizer_conformance(
    device: str | torch.device = "cpu",
    *,
    checkpoints: Sequence[int] = (1, 16, 64),
    max_parameter_absolute_error: float = 2e-8,
    max_loss_absolute_error: float = 1e-7,
) -> SplitRealImagOptimizerConformanceReport:
    """Compare P5 Double-Single and FP32-master SGD with complex128 trajectories."""

    resolved = resolve_platform_device(device)
    normalized_checkpoints = tuple(sorted({int(step) for step in checkpoints}))
    if not normalized_checkpoints or normalized_checkpoints[0] < 1:
        raise ValueError("P5 optimizer checkpoints must be positive")
    cancellation_ds, cancellation_fp32 = _cancellation_case(
        max(normalized_checkpoints), device=resolved
    )
    cases: list[SplitRealImagOptimizerTrajectoryCase] = []
    for index, (name, workload) in enumerate(
        (("two_qubit_vqe", _vqe_workload()), ("three_qubit_qaoa", _qaoa_workload()))
    ):
        circuit, observable, initial, learning_rate = workload
        cases.extend(
            _trajectory(
                name,
                circuit,
                observable,
                initial,
                learning_rate,
                checkpoints=normalized_checkpoints,
                max_parameter_absolute_error=max_parameter_absolute_error,
                max_loss_absolute_error=max_loss_absolute_error,
                preflight=index == 0,
                device=resolved,
            )
        )
    passed = bool(cases) and all(case.passed for case in cases)
    passed = passed and cancellation_ds <= 1e-14 and cancellation_fp32 > 0.0
    return SplitRealImagOptimizerConformanceReport(
        device=str(resolved),
        provider=get_platform_runtime(resolved.type).identity().provider,
        cases=tuple(cases),
        cancellation_double_single_absolute_error=cancellation_ds,
        cancellation_float32_absolute_error=cancellation_fp32,
        passed=passed,
        native_cuda_evidence=resolved.type == "cuda",
        torch_fl_flagos_evidence=resolved.type == "flagos",
    )


__all__ = (
    "SplitRealImagOptimizerConformanceReport",
    "SplitRealImagOptimizerTrajectoryCase",
    "run_split_real_imag_optimizer_conformance",
)
