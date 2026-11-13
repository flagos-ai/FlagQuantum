"""Review-only named angle binding; no production API is added by this fake.

The consumer uses the existing planner, codec and statevector engine. The two
binding implementations exercise Core substitution and Runtime parameter slots.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from math import isfinite
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.core.parameters import (
    ParameterExpression,
    bind_parameter_value,
    parameter_names_in_value,
)
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.runtime import execution_plan_contract as codec
from flagquantum.runtime._initial_state import (
    validate_input_decision,
    validate_input_program,
)
from flagquantum.runtime.builder_compilation import (
    BuilderBindings,
    SlotParameterMapping,
)
from flagquantum.runtime.options_resolver import (
    circuit_execution_constraints,
    resolve_execution_options,
)
from flagquantum.runtime.planner import plan_advanced

pytestmark = pytest.mark.unit

Bindings = Mapping[str, float | torch.Tensor]
Binder = Callable[[fq.CircuitIR, Bindings], fq.CircuitIR]


def _names(program: fq.CircuitIR) -> tuple[str, ...]:
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


def _extension(program: fq.CircuitIR) -> dict[str, object]:
    payload = {"parameters": list(_names(program))}
    return {
        "namespace": "flagquantum.execution",
        "kind": "runtime_inputs",
        "version": "1.0",
        "identity": codec.canonical_hash(payload),
        "payload": payload,
    }


def _seal(payload: dict[str, Any]) -> str:
    # Any stops at the existing JSON codec boundary.
    payload["identity"] = codec._identity(payload, compiler=codec._compiler_identity())
    return codec.canonical_json(payload)


def _review_payload(plan: fq.ExecutionPlan) -> dict[str, Any]:
    if plan._contract_payload_json is None:
        raise ValidationError("review requires a sealed plan")
    payload = json.loads(plan._contract_payload_json)
    if not isinstance(payload, dict):
        raise ValidationError("review requires a plan object")
    program = fq.CircuitIR.from_dict(payload["program"])
    if not _names(program) or payload.get("extensions") != [_extension(program)]:
        raise CapabilityError("review requires exactly one runtime-input extension")
    if payload.get("identity") != codec._identity(
        payload, compiler=codec._compiler_identity()
    ):
        raise ValidationError("review plan identity mismatch")
    # Keep the fake independent of production binding while delegating ordinary
    # schema and fingerprint validation to the authoritative plan codec.
    ordinary = dict(payload, extensions=[])
    _seal(ordinary)
    codec.validate_plan_payload(ordinary)
    validate_input_decision(program, payload["decision"], payload["resolved_options"])
    if _contains_tensor(tuple(item.params for item in program.instructions)):
        raise CapabilityError("review does not capture tensor literals")
    if codec._canonical_program(program, plan).content_hash != program.content_hash:
        raise ValidationError("review plan layers differ from the sealed template")
    return payload


def _review_plan(circuit: fq.Circuit, *, gradients: bool = False) -> fq.ExecutionPlan:
    if type(circuit) is not fq.Circuit or circuit.bsz != 1:
        raise CapabilityError("review requires an ordinary unbatched Circuit")
    if circuit.circuit_param.get("inputs") is not None:
        raise CapabilityError("review does not accept construction initial states")
    program = circuit.to_ir()
    validate_input_program(program)
    if not _names(program):
        raise ValidationError("runtime inputs require symbolic parameters")
    # Tensor literals would capture a previous run's storage/graph in the plan.
    # Symbolic expressions already have an authoritative serializable codec.
    if _contains_tensor(tuple(item.params for item in program.instructions)):
        raise CapabilityError("review does not capture tensor literals")
    requested = fq.ExecutionOptions(
        mode="statevector",
        backend="pytorch",
        device="cpu",
        require_gradients=gradients,
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    resolved = resolve_execution_options(
        requested, program_constraints=circuit_execution_constraints(circuit)
    )
    plan = plan_advanced(
        program,
        state_mode="statevector",
        complex_bytes=16 if resolved.precision == "complex128" else 8,
        require_gradients=gradients,
        allow_approximate=False,
        optimize=False,
    )
    plan = codec.attach_execution_contract(
        plan, program=program, requested_options=requested, resolved_options=resolved
    )
    payload = plan.to_dict()
    payload["extensions"] = [_extension(program)]
    plan = replace(plan, _contract_payload_json=_seal(payload))
    _review_payload(plan)
    return plan


def _restore_review_plan(text: str) -> fq.ExecutionPlan:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValidationError("review requires a plan object")
    if payload.get("identity") != codec._identity(
        payload, compiler=codec._compiler_identity()
    ):
        raise ValidationError("review plan identity mismatch")
    ordinary = dict(payload, extensions=[])
    _seal(ordinary)
    restored = codec.plan_from_dict(ordinary)
    restored = replace(restored, _contract_payload_json=codec.canonical_json(payload))
    _review_payload(restored)
    return restored


def _checked_inputs(plan: fq.ExecutionPlan, inputs: object) -> dict[str, torch.Tensor]:
    payload = _review_payload(plan)
    program = fq.CircuitIR.from_dict(payload["program"])
    if not isinstance(inputs, Mapping) or any(type(key) is not str for key in inputs):
        raise TypeError("inputs must be a mapping with string keys")
    if set(inputs) != set(_names(program)):
        raise ValidationError("inputs must exactly cover the symbolic parameter names")
    dtype = (
        torch.float64
        if payload["decision"]["precision"] == "complex128"
        else torch.float32
    )
    checked = {}
    for name, value in inputs.items():
        if type(value) is float:
            if not isfinite(value):
                raise ValidationError("runtime angles must be finite")
            tensor = torch.tensor(value, dtype=dtype)
        elif type(value) in (torch.Tensor, torch.nn.Parameter):
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
        if tensor.requires_grad and not payload["decision"]["require_gradients"]:
            raise CapabilityError("gradient inputs require a gradient-capable plan")
        checked[name] = tensor
    return checked


def _bind_core(program: fq.CircuitIR, inputs: Bindings) -> fq.CircuitIR:
    return fq.Circuit.from_ir(program).bind_parameters(inputs).to_ir()


def _bind_slots(program: fq.CircuitIR, inputs: Bindings) -> fq.CircuitIR:
    bindings = BuilderBindings()
    values: list[torch.Tensor] = []
    instructions = []
    try:
        for instruction in program.instructions:
            slots = {}
            for name, expression in instruction.params.items():
                value = bind_parameter_value(expression, inputs)
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(
                        value,
                        dtype=(
                            torch.float64
                            if program.dtype == "complex128"
                            else torch.float32
                        ),
                    )
                slots[name] = len(values)
                values.append(value)
            instructions.append((instruction, slots))
        bindings.bind(tuple(values))
        # Materialize one attempt's IR while the slots are bound. Clearing them
        # cannot alter this attempt or a graph retained by an earlier backward.
        return replace(
            program,
            instructions=tuple(
                replace(
                    instruction,
                    params=SlotParameterMapping({}, slots, bindings),
                )
                for instruction, slots in instructions
            ),
        )
    finally:
        bindings.clear()


def _execute_review(
    plan: fq.ExecutionPlan, inputs: object, binder: Binder = _bind_core
) -> torch.Tensor:
    checked = _checked_inputs(plan, inputs)
    payload = _review_payload(plan)
    program = binder(fq.CircuitIR.from_dict(payload["program"]), checked)
    # This is a numerical consumer, not fq.run acceptance of the review plan.
    state = fq.Circuit.from_ir(program).state()
    if not isinstance(state, torch.Tensor):
        raise TypeError("review statevector execution must return a Tensor")
    return state.reshape(-1)


@pytest.mark.parametrize("binder", [_bind_core, _bind_slots], ids=["core", "slots"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_same_consumer_reuses_template_and_preserves_gradients(
    binder: Binder, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    theta = fq.Parameter("theta")
    circuit = fq.Circuit(1, dtype=dtype).ry(0, 2 * theta + 0.1)
    plan = _review_plan(circuit, gradients=True)
    text = plan._contract_payload_json
    assert text is not None
    restored = _restore_review_plan(text)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("replay reached the compiler or planner")

    monkeypatch.setattr("flagquantum.runtime.planner.compile_program", forbidden)
    monkeypatch.setattr("flagquantum.runtime.planner.plan", forbidden)
    monkeypatch.setattr("flagquantum.runtime.planner.plan_advanced", forbidden)
    for candidate in (plan, restored):
        angles = [
            torch.tensor(value, dtype=real_dtype, requires_grad=True)
            for value in (0.0, 0.37, -0.81)
        ]
        # Keep multiple forward graphs live before any backward to detect stale
        # slot reuse, a shared binding table, or cross-call graph retention.
        outputs = [
            _execute_review(candidate, {"theta": value}, binder) for value in angles
        ]
        for angle, output in zip(angles, outputs, strict=True):
            half_angle = angle + 0.05
            expected = torch.stack((half_angle.cos(), half_angle.sin())).to(dtype)
            torch.testing.assert_close(output, expected)
            gradient = torch.autograd.grad(output[1].abs().square(), angle)[0]
            torch.testing.assert_close(gradient, (2 * half_angle).sin())
        assert candidate._contract_payload_json == text
    assert circuit.parameter_names == ("theta",)
    assert _checked_inputs(plan, {"theta": angles[0]})["theta"] is angles[0]


@pytest.mark.parametrize("binder", [_bind_core, _bind_slots], ids=["core", "slots"])
def test_optimizer_updates_reuse_one_plan(binder: Binder) -> None:
    plan = _review_plan(
        fq.Circuit(1, dtype=torch.complex128).ry(0, fq.Parameter("theta")),
        gradients=True,
    )
    angle = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float64))
    optimizer = torch.optim.SGD([angle], lr=0.2)
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        output = _execute_review(plan, {"theta": angle}, binder)
        loss = output[1].abs().square()
        losses.append(float(loss.detach()))
        torch.autograd.backward(loss)
        assert angle.grad is not None
        optimizer.step()
    assert losses[-1] < losses[0]


@pytest.mark.parametrize(
    "inputs,error",
    [
        (None, TypeError),
        ({1: 0.1}, TypeError),
        ({}, ValidationError),
        ({"theta": 0.1, "unknown": 0.2}, ValidationError),
        ({"theta": True}, TypeError),
        ({"theta": 1}, TypeError),
        ({"theta": 0.2j}, TypeError),
        ({"theta": float("nan")}, ValidationError),
        ({"theta": float("inf")}, ValidationError),
        ({"theta": torch.tensor([0.2], dtype=torch.float64)}, ValidationError),
        ({"theta": torch.tensor(0.2, dtype=torch.float32)}, ValidationError),
        ({"theta": torch.tensor(0.2j, dtype=torch.complex128)}, ValidationError),
        ({"theta": torch.tensor(float("nan"), dtype=torch.float64)}, ValidationError),
        ({"theta": torch.tensor(0.2, device="meta")}, CapabilityError),
    ],
)
def test_invalid_inputs_fail_before_binding(
    inputs: object, error: type[Exception]
) -> None:
    plan = _review_plan(
        fq.Circuit(1, dtype=torch.complex128).ry(0, fq.Parameter("theta"))
    )

    def forbidden(program: fq.CircuitIR, values: Bindings) -> fq.CircuitIR:
        raise AssertionError("invalid inputs reached the binder")

    with pytest.raises(error):
        _execute_review(plan, inputs, forbidden)


def test_gradient_requirement_and_no_implicit_defaults() -> None:
    plan = _review_plan(
        fq.Circuit(1, dtype=torch.complex128).ry(0, fq.Parameter("theta"))
    )
    angle = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    with pytest.raises(CapabilityError, match="gradient-capable"):
        _execute_review(plan, {"theta": angle})
    torch.testing.assert_close(
        _execute_review(plan, {"theta": 0.0}),
        torch.tensor([1, 0], dtype=torch.complex128),
    )
    with pytest.raises(ValidationError, match="exactly cover"):
        _execute_review(plan, {})


def test_float_conversion_overflow_and_expression_overflow_are_rejected() -> None:
    plan = _review_plan(fq.Circuit(1).ry(0, fq.Parameter("theta")))
    with pytest.raises(ValidationError, match="finite"):
        _execute_review(plan, {"theta": 1e300})
    plan = _review_plan(
        fq.Circuit(1, dtype=torch.complex128).ry(0, 2 * fq.Parameter("theta"))
    )
    with pytest.raises(ValidationError, match="finite"):
        _execute_review(plan, {"theta": 1e308})


def test_unsupported_sources_and_tensor_capture_are_rejected() -> None:
    theta = fq.Parameter("theta")
    with pytest.raises(ValidationError, match="symbolic"):
        _review_plan(fq.Circuit(1).h(0))
    with pytest.raises(CapabilityError, match="unbatched"):
        _review_plan(fq.Circuit(1, bsz=2).ry(0, theta))
    with pytest.raises(CapabilityError, match="construction"):
        _review_plan(fq.Circuit(1, inputs=torch.tensor([1, 0])).ry(0, theta))
    with pytest.raises(CapabilityError, match="tensor literals"):
        _review_plan(fq.Circuit(1).ry(0, theta + torch.tensor(0.1)))


def test_admitted_reader_preserves_requirement_and_tampering_is_detected() -> None:
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, fq.Parameter("theta"))
    plan = _review_plan(circuit)
    text = plan._contract_payload_json
    assert text is not None
    assert fq.ExecutionPlan.from_json(text).to_json() == text
    unknown = json.loads(text)
    unknown["extensions"][0]["version"] = "unsupported"
    with pytest.raises(
        codec.ExecutionPlanContractError, match="extension_incompatible"
    ):
        fq.ExecutionPlan.from_json(codec.canonical_json(unknown))
    with pytest.raises(CapabilityError, match="runtime-input"):
        _execute_review(fq.plan(circuit), {"theta": 0.1})
    payload = json.loads(text)
    payload["extensions"][0]["payload"]["parameters"] = ["other"]
    with pytest.raises(CapabilityError, match="runtime-input"):
        _restore_review_plan(_seal(payload))
    payload = json.loads(text)
    payload["decision"]["require_gradients"] = True
    with pytest.raises(ValidationError, match="identity mismatch"):
        _restore_review_plan(codec.canonical_json(payload))
    stale = replace(plan, layers=fq.plan(circuit.x(0)).layers)
    with pytest.raises(ValidationError, match="layers differ"):
        _execute_review(stale, {"theta": 0.1})


@pytest.mark.parametrize("binder", [_bind_core, _bind_slots], ids=["core", "slots"])
def test_two_parameter_entangled_workflow_matches_dense_reference(
    binder: Binder,
) -> None:
    a, b = fq.Parameter("a"), fq.Parameter("b")
    circuit = fq.Circuit(2, dtype=torch.complex128).ry(0, a).cx(0, 1).rz(1, b)
    plan = _review_plan(circuit, gradients=True)
    left = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    right = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
    output = _execute_review(plan, {"b": right, "a": left}, binder)
    zero = torch.zeros((), dtype=torch.complex128)
    expected = torch.stack(
        (
            (left / 2).cos() * torch.exp(-0.5j * right),
            zero,
            zero,
            (left / 2).sin() * torch.exp(0.5j * right),
        )
    )
    torch.testing.assert_close(output, expected)
    objective = output.real.sum()
    torch.testing.assert_close(
        torch.autograd.grad(objective, (left, right), retain_graph=True),
        torch.autograd.grad(expected.real.sum(), (left, right)),
    )
