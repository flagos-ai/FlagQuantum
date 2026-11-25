"""Bind named scalar angles per attempt without retaining them in a plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

import torch

from ..core.ir import CircuitIR, MeasurementNode
from ..core.operator_schema import get_operator_schema
from ..core.parameters import (
    Parameter,
    ParameterExpression,
    parameter_names_in_value,
)
from ..errors import CapabilityError, ValidationError
from ._initial_state import validate_input_decision, validate_input_program
from .builder_compilation import _bind_named_instructions
from .options import ExecutionOptions
from .options_resolver import circuit_execution_constraints, resolve_execution_options

if TYPE_CHECKING:
    from .execution_plan import ExecutionPlan


def parameter_names(program: CircuitIR) -> tuple[str, ...]:
    return parameter_names_in_value(tuple(item.params for item in program.instructions))


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, ParameterExpression):
        return _contains_tensor(value.args)
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def validate_runtime_program(program: CircuitIR) -> None:
    """Keep named binding within the existing static unitary input boundary."""
    try:
        validate_input_program(program)
    except CapabilityError as error:
        raise CapabilityError(
            "runtime inputs require ordinary static unitary gates and state/probability outputs"
        ) from error
    if not parameter_names(program):
        raise ValidationError("runtime inputs require declared symbolic parameters")
    if _contains_tensor(
        (program.metadata, tuple(item.params for item in program.instructions))
    ) or any(_contains_tensor(item.metadata) for item in program.instructions):
        raise CapabilityError("runtime input templates cannot capture tensor literals")
    for instruction in program.instructions:
        schema = get_operator_schema(instruction.name)
        if schema is None or set(instruction.params) != set(schema.parameters):
            raise CapabilityError(
                "runtime inputs require exactly the declared gate parameters"
            )
        if any(not _scalar_expression(value) for value in instruction.params.values()):
            raise CapabilityError(
                "runtime input templates require scalar angle expressions"
            )


def _scalar_expression(value: object) -> bool:
    if isinstance(value, ParameterExpression):
        return all(_scalar_expression(item) for item in value.args)
    return isinstance(value, Parameter) or (
        isinstance(value, Real) and not isinstance(value, bool)
    )


def validate_runtime_decision(
    program: CircuitIR, decision: Mapping[str, object], resolved: Mapping[str, object]
) -> None:
    validate_runtime_program(program)
    try:
        validate_input_decision(program, decision, resolved)
    except CapabilityError as error:
        raise CapabilityError(
            "runtime inputs require exact unbatched CPU PyTorch statevector execution"
        ) from error
    limit = decision["memory_limit_bytes"]
    state_bytes = decision["state_bytes"]
    if isinstance(limit, int) and isinstance(state_bytes, int) and state_bytes > limit:
        raise CapabilityError("runtime inputs exceed the state memory budget")


def _input_mapping(inputs: object) -> Mapping[str, object]:
    if inputs is None:
        raise ValidationError("this ExecutionPlan requires inputs on every execution")
    if not isinstance(inputs, Mapping) or any(type(key) is not str for key in inputs):
        raise TypeError("inputs must be a mapping with string keys")
    return inputs


def _checked_inputs(
    program: CircuitIR, inputs: object, *, precision: str, gradients: bool
) -> dict[str, torch.Tensor]:
    values = _input_mapping(inputs)
    if set(values) != set(parameter_names(program)):
        raise ValidationError("inputs must exactly cover the symbolic parameter names")
    dtype = torch.float64 if precision == "complex128" else torch.float32
    checked: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if type(value) is float:
            if not isfinite(value):
                raise ValidationError("runtime angles must be finite")
            tensor = torch.tensor(value, dtype=dtype)
        elif isinstance(value, torch.Tensor) and type(value) in (
            torch.Tensor,
            torch.nn.Parameter,
        ):
            tensor = value
        else:
            raise TypeError("runtime angles must be Python floats or scalar tensors")
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise CapabilityError("runtime angles require dense strided CPU storage")
        if tensor.shape != () or tensor.dtype != dtype:
            raise ValidationError(
                "runtime angle shape or precision does not match plan"
            )
        if not bool(torch.isfinite(tensor.detach())):
            raise ValidationError("runtime angles must be finite")
        if tensor.requires_grad and not gradients:
            raise CapabilityError("gradient inputs require a gradient-capable plan")
        checked[name] = tensor
    return checked


def _bind_program(program: CircuitIR, values: Mapping[str, torch.Tensor]) -> CircuitIR:
    return replace(
        program,
        instructions=_bind_named_instructions(program.instructions, values),
    )


def plan_runtime_inputs(
    circuit: object,
    inputs: object,
    *,
    options: ExecutionOptions | None,
    measurements: Sequence[MeasurementNode] | None,
) -> ExecutionPlan:
    """Prepare a symbolic plan once using the existing resolver and planner."""
    from ..circuit import Circuit
    from .distributed.backend_policy import resolve_distributed_backend_policy
    from .execution_plan_contract import (
        _compiler_identity,
        _identity,
        attach_execution_contract,
        canonical_json,
        runtime_inputs_extension,
    )
    from .planner import _validated_plan_program, plan_advanced

    if type(circuit) is not Circuit:
        raise CapabilityError(
            "runtime inputs currently accept ordinary Circuit sources only"
        )
    if circuit.circuit_param.get("inputs") is not None:
        raise CapabilityError(
            "runtime inputs cannot use Circuit construction initial states"
        )
    if options is not None and not isinstance(options, ExecutionOptions):
        raise TypeError("options must be an ExecutionOptions or None")
    source = _validated_plan_program(
        circuit, measurements=measurements, noise_model=None
    )
    validate_runtime_program(source)
    values = _input_mapping(inputs)
    gradient_needed = any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in values.values()
    )
    if gradient_needed and options is not None and options.require_gradients is False:
        raise ValidationError(
            "require_gradients=False conflicts with gradient-bearing inputs"
        )
    resolved = resolve_execution_options(
        options,
        policy_options=ExecutionOptions(require_gradients=gradient_needed),
        program_constraints=circuit_execution_constraints(circuit),
    )
    if (
        resolved.mode not in ("auto", "statevector")
        or resolved.backend not in ("auto", "pytorch")
        or resolved.device not in ("auto", "cpu")
        or resolved.target not in ("auto", "state")
        or resolved.batch_size != 1
        or resolved.shots is not None
        or resolved.allow_approximate
        or resolved.allow_backend_fallback
        or resolve_distributed_backend_policy().effective_world_size != 1
    ):
        raise CapabilityError(
            "runtime inputs require exact unbatched CPU PyTorch statevector execution"
        )
    checked = _checked_inputs(
        source,
        values,
        precision=resolved.precision,
        gradients=resolved.require_gradients,
    )
    _bind_program(source, checked)
    minimum_bytes = (2**source.n_wires) * (
        16 if resolved.precision == "complex128" else 8
    )
    if (
        resolved.memory_limit_bytes is not None
        and minimum_bytes > resolved.memory_limit_bytes
    ):
        raise CapabilityError("runtime inputs exceed the state memory budget")
    internal = plan_advanced(
        source,
        complex_bytes=16 if resolved.precision == "complex128" else 8,
        memory_limit_bytes=resolved.memory_limit_bytes,
        state_mode="statevector",
        require_gradients=resolved.require_gradients,
        allow_approximate=False,
        optimize=False,
    )
    if internal.analysis is None or internal.analysis.n_qubits != source.n_wires:
        raise CapabilityError("runtime input planning cannot introduce work qubits")
    plan = attach_execution_contract(
        internal,
        program=source,
        requested_options=options or ExecutionOptions(),
        resolved_options=resolved,
    )
    payload = plan.to_dict()
    payload["extensions"] = [runtime_inputs_extension(source)]
    payload["identity"] = _identity(payload, compiler=_compiler_identity())
    return replace(plan, _contract_payload_json=canonical_json(payload))


def bind_runtime_inputs(plan: ExecutionPlan, inputs: object) -> CircuitIR:
    """Return one attempt's numerical IR, preserving tensors and the sealed plan."""
    from .execution_plan_contract import (
        ExecutionPlanContractError,
        has_runtime_inputs_extension,
        plan_execution_program,
    )

    payload = plan.to_dict()
    if not has_runtime_inputs_extension(payload):
        raise CapabilityError("this plan does not accept runtime inputs")
    program = plan_execution_program(plan)
    if program.content_hash != plan.program_fingerprint:
        raise ExecutionPlanContractError(
            "program_fingerprint_mismatch",
            "live layers differ from the sealed symbolic template",
        )
    decision = payload["decision"]
    if not isinstance(decision, Mapping):  # validated by the plan codec
        raise ValidationError("runtime input plan decision must be a mapping")
    if plan.state_bytes != decision["state_bytes"]:
        raise ValidationError(
            "runtime input plan state_bytes differs from its sealed decision"
        )
    checked = _checked_inputs(
        program,
        inputs,
        precision=plan.precision,
        gradients=bool(decision["require_gradients"]),
    )
    return _bind_program(program, checked)
