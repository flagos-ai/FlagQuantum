"""Admission and owned input binding for exact CPU statevector execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from flagquantum.core.operators.schema import get_operator_schema

from ..core.classical_control import is_classical_control_instruction
from ..core.ir import CircuitIR, MeasurementNode
from ..errors import CapabilityError, ValidationError
from .options import ExecutionOptions
from .options_resolver import circuit_execution_constraints, resolve_execution_options

if TYPE_CHECKING:
    from .execution_plan import ExecutionPlan


def validate_input_program(program: CircuitIR) -> None:
    """Admit only arbitrary-input-safe ordinary static unitary programs."""
    from .planner import carries_noise_channels

    if program.observables or carries_noise_channels(program):
        raise CapabilityError(
            "external initial_state requires an ordinary unitary program"
        )
    if any(request.kind != "probabilities" for request in program.measurements):
        raise CapabilityError(
            "external initial_state supports statevectors and probabilities only"
        )
    for instruction in program.instructions:
        schema = get_operator_schema(instruction.name)
        if (
            schema is None
            or not schema.unitary
            or instruction.matrix is not None
            or is_classical_control_instruction(instruction)
        ):
            raise CapabilityError(
                "external initial_state supports ordinary static unitary opcodes only"
            )


def validate_input_decision(
    program: CircuitIR, decision: Mapping[str, object], resolved: Mapping[str, object]
) -> None:
    """Reject even rehashed plans whose marker contradicts the admitted scope."""
    validate_input_program(program)
    if (
        decision["mode"] != "statevector"
        or decision["backend"] != "pytorch"
        or decision["device"] != "cpu"
        or decision["batch_size"] != 1
        or decision["world_size"] != 1
        or decision["target"] not in ("auto", "state")
        or decision["allow_approximate"] is not False
        or decision["allow_backend_fallback"] is not False
        or resolved["shots"] is not None
        or resolved["mode"] not in ("auto", "statevector")
        or resolved["backend"] not in ("auto", "pytorch")
        or resolved["device"] not in ("auto", "cpu")
        or program.metadata.get("batch_size", 1) != 1
    ):
        raise CapabilityError(
            "external initial_state requires exact unbatched CPU PyTorch statevector execution"
        )
    for name in (
        "batch_size",
        "precision",
        "target",
        "require_gradients",
        "allow_approximate",
        "allow_backend_fallback",
        "memory_limit_bytes",
    ):
        if decision[name] != resolved[name]:
            raise ValidationError(
                f"initial_state decision disagrees with resolved {name}"
            )
    expected_bytes = (2**program.n_wires) * (
        16 if decision["precision"] == "complex128" else 8
    )
    if decision["state_bytes"] != expected_bytes:
        raise ValidationError(
            "initial_state plan must account for the complete register"
        )


def _input_tensor(state: object) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        raise TypeError("initial_state must be a torch.Tensor or torch.nn.Parameter")
    if type(state) not in (torch.Tensor, torch.nn.Parameter):
        raise CapabilityError("initial_state tensor subclasses are unsupported")
    if state.layout != torch.strided or state.device.type != "cpu":
        raise CapabilityError("initial_state requires dense strided CPU storage")
    return state


def _validate_tensor(state: object, *, n_qubits: int, precision: str) -> torch.Tensor:
    state = _input_tensor(state)
    if state.shape != (2**n_qubits,):
        raise ValidationError("initial_state shape must be (2**n_qubits,)")
    if state.dtype != getattr(torch, precision):
        raise ValidationError(
            "initial_state dtype must exactly match resolved precision"
        )
    return state


def _check_input_memory(
    state_bytes: int, tensor: torch.Tensor, limit: int | None
) -> None:
    if (
        limit is not None
        and state_bytes + tensor.numel() * tensor.element_size() > limit
    ):
        raise CapabilityError(
            "initial_state exceeds state and input-copy memory budget"
        )


def plan_initial_state(
    circuit: object,
    state: object,
    *,
    options: ExecutionOptions | None,
    measurements: Sequence[MeasurementNode] | None,
) -> ExecutionPlan:
    """Build an arbitrary-input plan using the existing planner and resolver."""
    from ..circuit import Circuit
    from .distributed.backend_policy import resolve_distributed_backend_policy
    from .execution_plan_contract import (
        _compiler_identity,
        _identity,
        attach_execution_contract,
        canonical_json,
        initial_state_extension,
    )
    from .planner import _validated_plan_program, plan_advanced

    if type(circuit) is not Circuit:
        raise CapabilityError(
            "external initial_state currently accepts ordinary Circuit sources only"
        )
    if circuit.circuit_param.get("inputs") is not None:
        raise TypeError("initial_state conflicts with Circuit construction inputs")
    if options is not None and not isinstance(options, ExecutionOptions):
        raise TypeError("options must be an ExecutionOptions or None")
    tensor = _input_tensor(state)
    source = _validated_plan_program(
        circuit, measurements=measurements, noise_model=None
    )
    validate_input_program(source)
    gradient_needed = tensor.requires_grad or any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for instruction in source.instructions
        for value in instruction.params.values()
    )
    if gradient_needed and options is not None and options.require_gradients is False:
        raise ValidationError(
            "require_gradients=False conflicts with gradient-bearing inputs or parameters"
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
            "external initial_state requires exact unbatched CPU PyTorch statevector execution"
        )
    _validate_tensor(state, n_qubits=source.n_wires, precision=resolved.precision)
    _check_input_memory(
        tensor.numel() * tensor.element_size(), tensor, resolved.memory_limit_bytes
    )
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
        raise CapabilityError("external initial_state cannot introduce work qubits")
    plan = attach_execution_contract(
        internal,
        program=source,
        requested_options=options or ExecutionOptions(),
        resolved_options=resolved,
    )
    payload = plan.to_dict()
    payload["extensions"] = [initial_state_extension()]
    payload["identity"] = _identity(payload, compiler=_compiler_identity())
    return replace(plan, _contract_payload_json=canonical_json(payload))


def bind_initial_state(plan: ExecutionPlan, state: object) -> torch.Tensor:
    """Validate per-call data, preflight the minimum budget, then own one copy."""
    from .execution_plan_contract import (
        ExecutionPlanContractError,
        has_initial_state_extension,
        plan_execution_program,
    )

    if not has_initial_state_extension(plan.to_dict()):
        raise CapabilityError("this plan does not accept external initial_state")
    if state is None:
        raise ValidationError(
            "this ExecutionPlan requires initial_state on every execution"
        )
    program = plan_execution_program(plan)
    if program.content_hash != plan.program_fingerprint:
        raise ExecutionPlanContractError(
            "program_fingerprint_mismatch",
            "gate bindings changed after planning; run the source Circuit to create a new plan",
        )
    tensor = _validate_tensor(state, n_qubits=program.n_wires, precision=plan.precision)
    decision = plan.to_dict()["decision"]
    if not isinstance(decision, Mapping):  # validated by the existing plan codec
        raise ValidationError("initial_state plan decision must be a mapping")
    if plan.state_bytes != decision["state_bytes"]:
        raise ValidationError(
            "initial_state plan state_bytes differs from its sealed decision"
        )
    if tensor.requires_grad and not decision["require_gradients"]:
        raise CapabilityError(
            "gradient-bearing initial_state requires a gradient-capable plan"
        )
    limit = decision["memory_limit_bytes"]
    _check_input_memory(
        plan.state_bytes, tensor, limit if isinstance(limit, int) else None
    )
    squared_norm = torch.zeros((), dtype=torch.float64)
    detached = tensor.detach()
    for offset in range(0, tensor.numel(), 4096):
        chunk = detached[offset : offset + 4096]
        if not bool(torch.isfinite(chunk).all()):
            raise ValidationError("initial_state must be finite")
        squared_norm += chunk.abs().to(torch.float64).square().sum()
    tolerance = 1e-12 if tensor.dtype == torch.complex128 else 1e-5
    if abs(float(squared_norm) - 1.0) > tolerance:
        raise ValidationError(
            "initial_state must have unit squared norm; normalization is not implicit"
        )
    return tensor.clone(memory_format=torch.contiguous_format)
