"""P5 precision-preserving SGD over explicit Double-Single gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ....core.numerics import AccuracyRequirementContract, PrecisionPlanContract
from ....core.parameters import Parameter
from ....simulation.numerics.double_single import (
    DoubleSingleTensor,
    _require_device_true,
)
from .split_real_imag import _canonical_parameter_bindings
from .split_real_imag_device_double_single import (
    SplitRealImagDeviceDoubleSingleGradientResult,
    parameter_shift_split_real_imag_device_double_single_gradient,
)

P5_OPTIMIZER = "split_real_imag_statevector_p5_double_single_sgd"


def _scalar_pair(value: Any, *, name: str) -> DoubleSingleTensor:
    if isinstance(value, DoubleSingleTensor):
        if value.high.numel() != 1:
            raise ValueError(f"P5 optimizer parameter {name!r} must be a scalar")
        return DoubleSingleTensor(
            value.high.detach().reshape(()).clone(),
            value.low.detach().reshape(()).clone(),
        ).renormalized()
    if not isinstance(value, torch.Tensor):
        raise TypeError("P5 optimizer parameters must be PyTorch tensors or pairs")
    if value.numel() != 1 or value.is_complex():
        raise ValueError(f"P5 optimizer parameter {name!r} must be a real scalar")
    if value.dtype != torch.float32:
        raise TypeError("P5 optimizer parameter tensors must use torch.float32")
    return DoubleSingleTensor.from_float32(value.detach().reshape(()).clone())


@dataclass(frozen=True)
class SplitRealImagDoubleSingleSGDState:
    """Named high/low master parameters updated by returning a new state."""

    parameter_order: tuple[str, ...]
    parameters: DoubleSingleTensor
    step: int = 0

    def __post_init__(self) -> None:
        if not self.parameter_order or len(set(self.parameter_order)) != len(
            self.parameter_order
        ):
            raise ValueError(
                "P5 optimizer parameter order must be non-empty and unique"
            )
        if tuple(sorted(self.parameter_order)) != self.parameter_order:
            raise ValueError("P5 optimizer parameter order must be canonical")
        if self.parameters.high.shape != (len(self.parameter_order),):
            raise ValueError("P5 optimizer parameter words must match parameter order")
        if self.step < 0:
            raise ValueError("P5 optimizer step must be non-negative")

    @property
    def device(self) -> torch.device:
        return self.parameters.high.device

    def bindings(self) -> dict[str, DoubleSingleTensor]:
        return {
            name: DoubleSingleTensor(
                self.parameters.high[index], self.parameters.low[index]
            )
            for index, name in enumerate(self.parameter_order)
        }

    def cpu_float64(self) -> torch.Tensor:
        """Reconstruct master parameters for diagnostics only."""

        return self.parameters.to("cpu").to_float64().detach()

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_split_real_imag_p5_sgd_state_v1",
            "optimizer": P5_OPTIMIZER,
            "algorithm": "double_single_sgd_without_momentum",
            "parameter_order": self.parameter_order,
            "parameter_representation": "double_single_high_low",
            "parameter_word_dtype": "float32",
            "parameter_word_count": 2,
            "step": self.step,
            "device": str(self.device),
            "distribution_semantics": "single_device_fast_path",
            "torch_optimizer_compatible": False,
            "optimizer_state_dict_compatible": False,
            "native_cuda_evidence": True,
            "torch_fl_flagos_evidence": True,
            "accelerator_float64_tensor_materialized": False,
            "training_state_device_resident": True,
            "per_step_host_tensor_transfer": False,
            "flagcx_collectives_validated": False,
            "convergence_certification": False,
            "production_claim_allowed": False,
        }


@dataclass(frozen=True)
class SplitRealImagDoubleSingleSGDStepResult:
    """One explicit P4-gradient/P5-optimizer step."""

    state: SplitRealImagDoubleSingleSGDState
    gradient_result: SplitRealImagDeviceDoubleSingleGradientResult
    learning_rate: DoubleSingleTensor

    def summary(self) -> dict[str, Any]:
        summary = self.state.summary()
        summary.update(
            {
                "schema": "flagquantum_split_real_imag_p5_sgd_step_v1",
                "gradient_representation": "double_single_high_low",
                "gradient_source": "p4_explicit_parameter_shift",
                "tensor_grad_used": False,
                "learning_rate_representation": "double_single_high_low",
            }
        )
        return summary


def initialize_split_real_imag_double_single_sgd(
    parameter_bindings: Mapping[str | Parameter, torch.Tensor | DoubleSingleTensor],
) -> SplitRealImagDoubleSingleSGDState:
    """Create owned co-resident high/low masters from direct named bindings."""

    normalized: dict[str, DoubleSingleTensor] = {}
    for name, value in _canonical_parameter_bindings(parameter_bindings).items():
        normalized[name] = _scalar_pair(value, name=name)
    order = tuple(sorted(normalized))
    if not order:
        raise ValueError("P5 optimizer requires at least one named parameter")
    return SplitRealImagDoubleSingleSGDState(
        parameter_order=order,
        parameters=DoubleSingleTensor(
            torch.stack(tuple(normalized[name].high for name in order)),
            torch.stack(tuple(normalized[name].low for name in order)),
        ).renormalized(),
    )


def _learning_rate_pair(
    value: torch.Tensor | DoubleSingleTensor,
    *,
    like: torch.Tensor,
) -> DoubleSingleTensor:
    if isinstance(value, DoubleSingleTensor):
        pair = value
        if pair.high.numel() != 1 or pair.high.device != like.device:
            raise ValueError(
                "P5 learning-rate pair must be a scalar on the state device"
            )
    elif isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype != torch.float32 or value.is_complex():
            raise TypeError("P5 learning-rate tensor must be a real FP32 scalar")
        if value.device != like.device:
            raise ValueError(
                "P5 learning-rate tensor must already be on the state device"
            )
        pair = DoubleSingleTensor.from_float32(value.detach().reshape(()))
    else:
        raise TypeError(
            "P5 learning rate must be a device-resident FP32 tensor or "
            "DoubleSingleTensor"
        )
    value = pair.to_float32()
    _require_device_true(
        torch.isfinite(value) & (value > 0),
        message="P5 learning rate must be finite and strictly positive",
    )
    return pair.renormalized()


def double_single_sgd_step(
    state: SplitRealImagDoubleSingleSGDState,
    gradient: DoubleSingleTensor,
    *,
    parameter_order: Sequence[str],
    learning_rate: torch.Tensor | DoubleSingleTensor,
) -> tuple[SplitRealImagDoubleSingleSGDState, DoubleSingleTensor]:
    """Apply one renormalized high/low SGD update without using ``Tensor.grad``."""

    if tuple(parameter_order) != state.parameter_order:
        raise ValueError("P5 optimizer gradient parameter order does not match state")
    if gradient.high.shape != state.parameters.high.shape:
        raise ValueError("P5 optimizer gradient shape does not match state")
    if gradient.high.device != state.device:
        raise ValueError("P5 optimizer gradient must reside on the state device")
    rate = _learning_rate_pair(learning_rate, like=state.parameters.high)
    expanded_rate = DoubleSingleTensor(
        rate.high.expand_as(gradient.high), rate.low.expand_as(gradient.low)
    )
    next_parameters = state.parameters.subtract(
        gradient.multiply(expanded_rate)
    ).renormalized()
    _require_device_true(
        torch.isfinite(next_parameters.high).all()
        & torch.isfinite(next_parameters.low).all(),
        message="P5 optimizer update produced a non-finite parameter",
        error_type=FloatingPointError,
    )
    return (
        SplitRealImagDoubleSingleSGDState(
            parameter_order=state.parameter_order,
            parameters=next_parameters,
            step=state.step + 1,
        ),
        rate,
    )


def split_real_imag_double_single_sgd_step(
    circuit_or_ir: Any,
    observable: Any | None,
    state: SplitRealImagDoubleSingleSGDState,
    *,
    learning_rate: torch.Tensor | DoubleSingleTensor,
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None = None,
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None = None,
    preflight: bool = True,
    renormalize_every: int = 16,
) -> SplitRealImagDoubleSingleSGDStepResult:
    """Evaluate an explicit P4 gradient and apply one Double-Single SGD step."""

    gradient_result = parameter_shift_split_real_imag_device_double_single_gradient(
        circuit_or_ir,
        observable,
        parameter_bindings=state.bindings(),
        device=state.device,
        precision_plan=precision_plan,
        accuracy_requirement=accuracy_requirement,
        preflight=preflight,
        renormalize_every=renormalize_every,
    )
    next_state, rate = double_single_sgd_step(
        state,
        gradient_result.gradient,
        parameter_order=gradient_result.parameter_order,
        learning_rate=learning_rate,
    )
    return SplitRealImagDoubleSingleSGDStepResult(
        state=next_state,
        gradient_result=gradient_result,
        learning_rate=rate,
    )


__all__ = (
    "SplitRealImagDoubleSingleSGDState",
    "SplitRealImagDoubleSingleSGDStepResult",
    "double_single_sgd_step",
    "initialize_split_real_imag_double_single_sgd",
    "split_real_imag_double_single_sgd_step",
)
