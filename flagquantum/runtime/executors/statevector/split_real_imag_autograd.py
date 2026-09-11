"""CPU P5 PyTorch autograd bridge over P4 Double-Single execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from ....compute import get_platform_runtime, resolve_platform_device
from ....core.ir import ensure_circuit_ir
from ....core.numerics import AccuracyRequirementContract, PrecisionPlanContract
from ....core.parameters import Parameter
from .split_real_imag import _parameter_occurrences
from .split_real_imag_device_double_single import (
    execute_split_real_imag_device_double_single_expectation,
    parameter_shift_split_real_imag_device_double_single_gradient,
)

P5_EXECUTOR = "split_real_imag_statevector_p5_autograd_bridge"


@dataclass(frozen=True)
class _P5BridgeConfig:
    circuit_or_ir: Any
    observable: Any
    parameter_order: tuple[str, ...]
    device: torch.device
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None
    preflight: bool
    renormalize_every: int


def split_real_imag_p5_autograd_bridge_summary() -> dict[str, Any]:
    """Return the static, machine-readable boundary for the CPU P5 bridge."""

    return {
        "schema": "flagquantum_split_real_imag_p5_autograd_bridge_v1",
        "executor": P5_EXECUTOR,
        "maturity": "experimental",
        "phase_status": "cpu_autograd_bridge",
        "distribution_semantics": "single_device_fast_path",
        "forward_representation": "double_single_fp32_complex",
        "backward_method": "parameter_shift",
        "internal_gradient_representation": "double_single_high_low",
        "delivered_tensor_grad_precision": "float32_boundary",
        "pytorch_tensor_grad_word_count": 1,
        "silent_dtype_demotion": False,
        "end_to_end_double_single_gradient_claim_allowed": False,
        "optimizer_available": True,
        "optimizer_boundary": "explicit_double_single_gradient_not_tensor_grad",
        "higher_order_autograd": False,
        "cpu_reference_evidence": True,
        "native_cuda_evidence": False,
        "torch_fl_flagos_evidence": False,
        "flagcx_collectives_validated": False,
        "convergence_certification": False,
        "hardware_certification": False,
        "scalability_claim_allowed": False,
        "performance_claim_allowed": False,
        "production_claim_allowed": False,
    }


def _canonical_parameters(
    parameter_bindings: Mapping[str | Parameter, torch.Tensor],
    *,
    expected: Sequence[str],
    device: torch.device,
) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
    normalized: dict[str, torch.Tensor] = {}
    for raw_name, value in parameter_bindings.items():
        name = raw_name.name if isinstance(raw_name, Parameter) else str(raw_name)
        if not name or name in normalized:
            raise ValueError(f"duplicate or empty P5 parameter binding {name!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError("P5 autograd bindings must be PyTorch tensors")
        if value.numel() != 1 or value.is_complex():
            raise ValueError("P5 autograd bindings must be real scalar tensors")
        if value.dtype != torch.float32:
            raise TypeError("P5 autograd bindings must use torch.float32")
        if value.device != device:
            raise ValueError("P5 autograd bindings must already reside on the CPU")
        if not value.requires_grad:
            raise ValueError("P5 autograd bindings must require gradients")
        normalized[name] = value.reshape(())
    order = tuple(sorted(expected))
    if set(normalized) != set(order) or not order:
        raise ValueError(
            "P5 bindings must exactly match at least one named parameter; "
            f"expected {list(order)}, got {sorted(normalized)}"
        )
    return order, tuple(normalized[name] for name in order)


class _P5AutogradContext(Protocol):
    config: _P5BridgeConfig

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _P5ParameterShiftExpectation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _P5AutogradContext,
        config: _P5BridgeConfig,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        bindings: dict[str | Parameter, torch.Tensor] = {
            name: parameter.detach()
            for name, parameter in zip(config.parameter_order, parameters)
        }
        result = execute_split_real_imag_device_double_single_expectation(
            config.circuit_or_ir,
            config.observable,
            parameter_bindings=bindings,
            device=config.device,
            precision_plan=config.precision_plan,
            accuracy_requirement=config.accuracy_requirement,
            preflight=config.preflight,
            renormalize_every=config.renormalize_every,
        )
        ctx.config = config
        ctx.save_for_backward(*(parameter.detach() for parameter in parameters))
        return result.value.to_float32().reshape(())

    @staticmethod
    def backward(
        ctx: _P5AutogradContext, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("P5 CPU bridge does not support higher-order autograd")
        config: _P5BridgeConfig = ctx.config
        bindings: dict[str | Parameter, torch.Tensor] = {
            name: parameter
            for name, parameter in zip(config.parameter_order, ctx.saved_tensors)
        }
        result = parameter_shift_split_real_imag_device_double_single_gradient(
            config.circuit_or_ir,
            config.observable,
            parameter_bindings=bindings,
            device=config.device,
            precision_plan=config.precision_plan,
            accuracy_requirement=config.accuracy_requirement,
            preflight=False,
            renormalize_every=config.renormalize_every,
        )
        if result.parameter_order != config.parameter_order:
            raise RuntimeError("P5 parameter order changed across autograd boundary")
        delivered = result.gradient.to_float32()
        upstream = grad_output.to(device=config.device, dtype=torch.float32).reshape(())
        return (None, *(upstream * delivered[index] for index in range(len(delivered))))


def split_real_imag_device_double_single_autograd_expectation(
    circuit_or_ir: Any,
    observable: Any | None = None,
    *,
    parameter_bindings: Mapping[str | Parameter, torch.Tensor],
    device: str | torch.device = "cpu",
    precision_plan: PrecisionPlanContract | Mapping[str, Any] | None = None,
    accuracy_requirement: AccuracyRequirementContract | Mapping[str, Any] | None = None,
    preflight: bool = True,
    renormalize_every: int = 16,
) -> torch.Tensor:
    """Return a scalar FP32 loss with a P4 parameter-shift backward bridge."""

    resolved_device = resolve_platform_device(device)
    if resolved_device.type != "cpu":
        raise NotImplementedError("the first P5 autograd bridge slice is CPU-only")
    ir = ensure_circuit_ir(circuit_or_ir)
    order, parameters = _canonical_parameters(
        parameter_bindings,
        expected=tuple(_parameter_occurrences(ir)),
        device=resolved_device,
    )
    identity = get_platform_runtime(resolved_device.type).identity()
    if identity.provider != "pytorch_cpu":
        raise RuntimeError("P5 CPU bridge resolved an unexpected compute runtime")
    config = _P5BridgeConfig(
        circuit_or_ir=circuit_or_ir,
        observable=observable,
        parameter_order=order,
        device=resolved_device,
        precision_plan=precision_plan,
        accuracy_requirement=accuracy_requirement,
        preflight=preflight,
        renormalize_every=renormalize_every,
    )
    # PyTorch's variadic apply boundary is untyped; validate its result here.
    apply: Callable[..., object] = _P5ParameterShiftExpectation.apply
    result = apply(config, *parameters)
    if not isinstance(result, torch.Tensor):
        raise TypeError("P5 autograd bridge must return a tensor")
    return result


__all__ = (
    "split_real_imag_device_double_single_autograd_expectation",
    "split_real_imag_p5_autograd_bridge_summary",
)
