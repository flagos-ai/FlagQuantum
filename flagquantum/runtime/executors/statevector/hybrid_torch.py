"""Private functional PyTorch operators for a specialized hybrid circuit region."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, Sequence

import torch

from ....core.ir import CircuitIR
from ....core.parameters import (
    Parameter,
    bind_parameter_value,
    parameter_names_in_value,
)
from .reverse import execute_torch_distributed_statevector_reverse

_PARAMETER_ORDER_KEY = "hybrid_parameter_order"
_SUPPORTED_REAL_DTYPES = (torch.float32, torch.float64)


def _decode_template(
    parameters: Sequence[torch.Tensor], circuit_ir_json: str
) -> tuple[CircuitIR, tuple[str, ...], tuple[int, ...]]:
    if not isinstance(circuit_ir_json, str):
        raise TypeError("circuit_ir_json must be a string")
    template = CircuitIR.from_json(circuit_ir_json)
    raw_order = template.metadata.get(_PARAMETER_ORDER_KEY)
    if not isinstance(raw_order, (tuple, list)) or not raw_order:
        raise ValueError(
            "hybrid CircuitIR must carry a non-empty hybrid_parameter_order"
        )
    names = tuple(str(name) for name in raw_order)
    if len(set(names)) != len(names):
        raise ValueError("hybrid_parameter_order must not contain duplicate names")
    discovered = {
        name
        for instruction in template.instructions
        for name in parameter_names_in_value(instruction.params)
    }
    if discovered != set(names):
        raise ValueError(
            "hybrid_parameter_order must exactly match the CircuitIR parameter slots"
        )
    if len(parameters) != len(names):
        raise ValueError(
            f"hybrid operator expected {len(names)} parameters, got {len(parameters)}"
        )
    if not parameters:
        raise ValueError("hybrid operator requires at least one parameter")
    first = parameters[0]
    if first.device.type != "cpu":
        raise ValueError("hybrid statevector operators support CPU tensors only")
    if first.dtype not in _SUPPORTED_REAL_DTYPES:
        raise TypeError("hybrid parameters must use float32 or float64")
    for index, parameter in enumerate(parameters):
        if parameter.ndim != 0:
            raise ValueError(f"hybrid parameter {index} must be a scalar tensor")
        if parameter.device != first.device or parameter.dtype != first.dtype:
            raise ValueError("hybrid parameters must share one dtype and device")
    expected_circuit_dtype = (
        "complex128" if first.dtype == torch.float64 else "complex64"
    )
    if template.dtype != expected_circuit_dtype:
        raise ValueError(
            f"{first.dtype} parameters require CircuitIR dtype {expected_circuit_dtype}"
        )
    if not template.observables:
        raise ValueError("hybrid operator requires at least one observable")
    for observable in template.observables:
        if (
            observable.name != "z"
            or len(observable.wires) != 1
            or observable.coefficient != 1.0
        ):
            raise ValueError(
                "hybrid statevector operators support sums of unit single-wire Z terms only"
            )
    return template, names, tuple(item.wires[0] for item in template.observables)


def _bind_template(
    template: CircuitIR, names: Sequence[str], parameters: Sequence[torch.Tensor]
) -> CircuitIR:
    bindings: dict[str | Parameter, torch.Tensor] = dict(zip(names, parameters))
    return replace(
        template,
        instructions=tuple(
            replace(
                instruction,
                params=bind_parameter_value(instruction.params, bindings),
            )
            for instruction in template.instructions
        ),
    )


@torch.library.custom_op(
    "flagquantum_hybrid::statevector_expectation_backward", mutates_args=()
)
def _statevector_expectation_backward_op(
    grad_output: torch.Tensor,
    parameters: list[torch.Tensor],
    circuit_ir_json: str,
) -> list[torch.Tensor]:
    template, names, observable_wires = _decode_template(parameters, circuit_ir_json)
    with torch.enable_grad():
        local_parameters = [
            parameter.detach().requires_grad_(True) for parameter in parameters
        ]
        circuit = _bind_template(template, names, local_parameters)
        result = execute_torch_distributed_statevector_reverse(
            circuit, observable_wires=observable_wires, device="cpu"
        )
        gradients = torch.autograd.grad(
            result.value,
            local_parameters,
            grad_outputs=grad_output,
            create_graph=False,
        )
    return [gradient.detach() for gradient in gradients]


@_statevector_expectation_backward_op.register_fake
def _statevector_expectation_backward_fake(
    grad_output: torch.Tensor,
    parameters: list[torch.Tensor],
    circuit_ir_json: str,
) -> list[torch.Tensor]:
    del grad_output, circuit_ir_json
    return [torch.empty_like(parameter) for parameter in parameters]


@torch.library.custom_op("flagquantum_hybrid::statevector_expectation", mutates_args=())
def _statevector_expectation_op(
    parameters: list[torch.Tensor], circuit_ir_json: str
) -> torch.Tensor:
    template, names, observable_wires = _decode_template(parameters, circuit_ir_json)
    local_parameters = [
        parameter.detach().requires_grad_(True) for parameter in parameters
    ]
    circuit = _bind_template(template, names, local_parameters)
    return execute_torch_distributed_statevector_reverse(
        circuit, observable_wires=observable_wires, device="cpu"
    ).value.detach()


@_statevector_expectation_op.register_fake
def _statevector_expectation_fake(
    parameters: list[torch.Tensor], circuit_ir_json: str
) -> torch.Tensor:
    del circuit_ir_json
    if not parameters:
        raise ValueError("hybrid operator requires at least one parameter")
    return parameters[0].new_empty(())


class _HybridAutogradContext(Protocol):
    circuit_ir_json: str

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _setup_context(
    ctx: _HybridAutogradContext,
    inputs: tuple[list[torch.Tensor], str],
    output: torch.Tensor,
) -> None:
    del output
    parameters, circuit_ir_json = inputs
    ctx.save_for_backward(*parameters)
    ctx.circuit_ir_json = circuit_ir_json


def _backward(
    context: _HybridAutogradContext, grad_output: torch.Tensor
) -> tuple[list[torch.Tensor], None]:
    gradients = _statevector_expectation_backward_op(
        grad_output, list(context.saved_tensors), context.circuit_ir_json
    )
    return gradients, None


_statevector_expectation_op.register_autograd(
    _backward,
    setup_context=_setup_context,
)


def hybrid_statevector_expectation(
    parameters: Sequence[torch.Tensor], circuit_ir_json: str
) -> torch.Tensor:
    """Execute one supported region through the private functional custom op."""

    result: object = _statevector_expectation_op(list(parameters), circuit_ir_json)
    if not isinstance(result, torch.Tensor):
        raise TypeError("hybrid statevector operator must return a tensor")
    return result


__all__ = ("hybrid_statevector_expectation",)
