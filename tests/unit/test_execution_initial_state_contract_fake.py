"""Review-only input binding retained for implementation replacement tests.

This fake reuses the planner, option resolver, plan codec and numerical engine.
It demonstrates the proposed contract, not a completed production replacement.
"""

import json
from dataclasses import replace
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.core import CircuitIR, Instruction, get_operator_schema
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.runtime import execution_plan_contract as codec
from flagquantum.runtime.options_resolver import (
    circuit_execution_constraints,
    resolve_execution_options,
)
from flagquantum.runtime.planner import plan_advanced

pytestmark = pytest.mark.unit


def _extension() -> dict[str, object]:
    payload = {"representation": "statevector"}
    return {
        "namespace": "flagquantum.execution",
        "kind": "initial_state",
        "version": "1.0",
        "identity": codec.canonical_hash(payload),
        "payload": payload,
    }


def _seal(payload: dict[str, Any]) -> str:
    # Any is confined to the existing JSON codec boundary.
    payload["identity"] = codec._identity(payload, compiler=codec._compiler_identity())
    return codec.canonical_json(payload)


def _review_payload(plan: fq.ExecutionPlan) -> dict[str, Any]:
    if plan._contract_payload_json is None:
        raise ValidationError("review requires a serialized plan")
    payload = json.loads(plan._contract_payload_json)
    if not isinstance(payload, dict):
        raise ValidationError("review requires a plan object")
    if payload.get("extensions") != [_extension()]:
        raise CapabilityError("review requires exactly one initial-state extension")
    if payload.get("identity") != codec._identity(
        payload, compiler=codec._compiler_identity()
    ):
        raise ValidationError("review plan identity mismatch")
    # Delegate all ordinary schema/fingerprint checks without relaxing production.
    ordinary = dict(payload, extensions=[])
    _seal(ordinary)
    codec.validate_plan_payload(ordinary)
    return payload


def _review_plan(
    circuit: fq.Circuit,
    *,
    requires_grad: bool = False,
    options: fq.ExecutionOptions | None = None,
) -> fq.ExecutionPlan:
    if circuit.circuit_param.get("inputs") is not None:
        raise TypeError("construction inputs conflict with external input")
    program = circuit.to_ir()
    if program.measurements or circuit.bsz != 1:
        raise CapabilityError("review supports unbatched static programs")
    for instruction in program.instructions:
        schema = get_operator_schema(instruction.name)
        if schema is None or not schema.unitary or instruction.matrix is not None:
            raise CapabilityError("review supports ordinary unitary opcodes only")
    gradient_needed = requires_grad or any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for instruction in program.instructions
        for value in instruction.params.values()
    )
    requested = options or fq.ExecutionOptions()
    if gradient_needed and requested.require_gradients is False:
        raise ValidationError("explicit gradient rejection conflicts with inputs")
    resolved = resolve_execution_options(
        requested,
        policy_options=fq.ExecutionOptions(require_gradients=gradient_needed),
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
    ):
        raise CapabilityError("review supports exact CPU statevector execution")
    plan = plan_advanced(
        program,
        complex_bytes=16 if resolved.precision == "complex128" else 8,
        state_mode="statevector",
        require_gradients=resolved.require_gradients,
        allow_approximate=False,
        optimize=False,
    )
    if plan.analysis is None or plan.analysis.n_qubits != circuit.n_qubits:
        raise CapabilityError("review does not admit new work qubits")
    plan = codec.attach_execution_contract(
        plan, program=program, requested_options=requested, resolved_options=resolved
    )
    payload = plan.to_dict()
    payload["extensions"] = [_extension()]
    return replace(plan, _contract_payload_json=_seal(payload))


def _restore_review_plan(text: str) -> fq.ExecutionPlan:
    payload = json.loads(text)
    candidate = fq.ExecutionPlan(
        analysis=None,
        layers=(),
        state_bytes=0,
        recommended_mode="statevector",
        world_size=1,
        shardable_wires=(),
        _contract_payload_json=codec.canonical_json(payload),
    )
    checked = _review_payload(candidate)
    ordinary = dict(checked, extensions=[])
    _seal(ordinary)
    restored = codec.plan_from_dict(ordinary)
    return replace(restored, _contract_payload_json=codec.canonical_json(checked))


def _bind_review_input(plan: fq.ExecutionPlan, state: object) -> torch.Tensor:
    payload = _review_payload(plan)
    if state is None:
        raise ValidationError("this plan requires initial_state on every execution")
    if not isinstance(state, torch.Tensor):
        raise TypeError("initial_state must be a Tensor")
    if type(state) not in (torch.Tensor, torch.nn.Parameter):
        raise CapabilityError("tensor subclasses require separate admission")
    if state.layout != torch.strided or state.device.type != "cpu":
        raise CapabilityError("review requires a dense strided CPU tensor")
    program = CircuitIR.from_dict(payload["program"])
    decision = payload["decision"]
    if state.shape != (2**program.n_wires,):
        raise ValidationError("initial_state shape must match the complete register")
    if state.dtype != getattr(torch, decision["precision"]):
        raise ValidationError("initial_state precision must match the plan")
    if state.requires_grad and not decision["require_gradients"]:
        raise CapabilityError("gradient input requires a gradient-capable plan")
    minimum = plan.state_bytes + state.numel() * state.element_size()
    limit = decision["memory_limit_bytes"]
    if limit is not None and limit < minimum:
        raise CapabilityError("insufficient state and input-copy memory budget")
    squared_norm = torch.zeros((), dtype=torch.float64)
    for chunk in torch.split(state.detach(), 4096):
        if not bool(torch.isfinite(chunk).all()):
            raise ValidationError("initial_state must be finite")
        squared_norm += chunk.abs().to(torch.float64).square().sum()
    tolerance = 1e-12 if state.dtype == torch.complex128 else 1e-5
    if abs(float(squared_norm) - 1.0) > tolerance:
        raise ValidationError("initial_state must have unit squared norm")
    return state.clone(memory_format=torch.contiguous_format)


def _execute_review(plan: fq.ExecutionPlan, state: object) -> torch.Tensor:
    owned = _bind_review_input(plan, state)
    payload = _review_payload(plan)
    # Preserve live parameters from plan layers, as the production codec does.
    program = codec._canonical_program(CircuitIR.from_dict(payload["program"]), plan)
    return torch.reshape(fq.Circuit.from_ir(program, inputs=owned).state(), (-1,))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_full_register_inputs_and_compatible_json_replay(dtype: torch.dtype) -> None:
    circuit = fq.Circuit(2, dtype=dtype).cx(0, 1).s(0)
    plan = _review_plan(circuit)
    text = plan._contract_payload_json
    assert text is not None
    restored = _restore_review_plan(text)
    # CX uses qubit 0 as MSB; S acts on both |10> and |11>.
    inputs_and_outputs = [
        ([0, 0, 0, 1], [0, 0, 1j, 0]),
        ([1 / 2**0.5, 0, 0, 1j / 2**0.5], [1 / 2**0.5, 0, -1 / 2**0.5, 0]),
    ]
    for source, expected in inputs_and_outputs:
        state = torch.tensor(source, dtype=dtype)
        for candidate in (plan, restored):
            actual = _execute_review(candidate, state)
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype))
            norm = actual.abs().square().sum()
            torch.testing.assert_close(norm, torch.ones_like(norm))
    assert restored._contract_payload_json == text
    with pytest.raises(ValidationError, match="requires initial_state"):
        _execute_review(restored, None)
    assert fq.ExecutionPlan.from_json(text).to_json() == text
    unknown = json.loads(text)
    unknown["extensions"][0]["version"] = "unsupported"
    with pytest.raises(
        codec.ExecutionPlanContractError, match="extension_incompatible"
    ):
        fq.ExecutionPlan.from_json(codec.canonical_json(unknown))


def test_input_and_gate_gradients_match_independent_dense_reference() -> None:
    state = torch.tensor([0.6, 0.8j], dtype=torch.complex128, requires_grad=True)
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    plan = _review_plan(
        fq.Circuit(1, dtype=torch.complex128).ry(0, angle), requires_grad=True
    )
    output = _execute_review(plan, state)
    loss = output[0].abs().square() + 0.3 * output[1].real
    gradients = torch.autograd.grad(loss, (state, angle))

    reference_state = state.detach().clone().requires_grad_()
    reference_angle = angle.detach().clone().requires_grad_()
    cosine, sine = (reference_angle / 2).cos(), (reference_angle / 2).sin()
    matrix = torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2)
    expected = matrix.to(torch.complex128) @ reference_state
    reference_loss = expected[0].abs().square() + 0.3 * expected[1].real
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        gradients,
        torch.autograd.grad(reference_loss, (reference_state, reference_angle)),
    )

    # Independent scalar central difference, with the input held fixed.
    def scalar_loss(theta: float) -> float:
        half = torch.tensor(theta / 2, dtype=torch.float64)
        first = half.cos() * state.detach()[0] - half.sin() * state.detach()[1]
        second = half.sin() * state.detach()[0] + half.cos() * state.detach()[1]
        return float(first.abs().square() + 0.3 * second.real)

    step = 1e-6
    numerical = (scalar_loss(0.37 + step) - scalar_loss(0.37 - step)) / (2 * step)
    assert float(gradients[1]) == pytest.approx(numerical, abs=1e-9)
    assert state.grad is None and angle.grad is None
    with pytest.raises(ValidationError, match="explicit gradient rejection"):
        _review_plan(
            fq.Circuit(1).ry(0, angle),
            options=fq.ExecutionOptions(require_gradients=False),
        )


def test_binding_owns_noncontiguous_storage_and_preserves_parameter_gradients() -> None:
    storage = torch.tensor([0.6, 99, 0.8j, 99], dtype=torch.complex128)
    state = torch.nn.Parameter(storage[::2])
    assert not state.is_contiguous()
    plan = _review_plan(fq.Circuit(1, dtype=torch.complex128), requires_grad=True)
    owned = _bind_review_input(plan, state)
    assert owned.is_contiguous() and owned.data_ptr() != state.data_ptr()
    torch.autograd.backward(owned.real.sum())
    torch.testing.assert_close(state.grad, torch.ones(2, dtype=torch.complex128))
    with torch.no_grad():
        state.zero_()
    torch.testing.assert_close(owned, torch.tensor([0.6, 0.8j], dtype=torch.complex128))
    assert _review_payload(plan)["extensions"] == [_extension()]


@pytest.mark.parametrize(
    "state, error",
    [
        ([1, 0], TypeError),
        (torch.ones(1, 2, dtype=torch.complex128), ValidationError),
        (torch.tensor([1, 0], dtype=torch.float64), ValidationError),
        (torch.tensor([1, 0], dtype=torch.complex64), ValidationError),
        (torch.zeros(2, dtype=torch.complex128), ValidationError),
        (torch.tensor([float("nan"), 0], dtype=torch.complex128), ValidationError),
        (torch.tensor([float("inf"), 0], dtype=torch.complex128), ValidationError),
        (torch.eye(2, dtype=torch.complex128).to_sparse(), CapabilityError),
    ],
)
def test_invalid_input_fails_without_coercion(
    state: object, error: type[Exception]
) -> None:
    plan = _review_plan(fq.Circuit(1, dtype=torch.complex128))
    with pytest.raises(error):
        _bind_review_input(plan, state)


def test_memory_rejection_precedes_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _review_plan(
        fq.Circuit(1, dtype=torch.complex128),
        options=fq.ExecutionOptions(memory_limit_bytes=63),
    )
    state = torch.tensor([0, 1], dtype=torch.complex128)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("input cloned before memory preflight")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    with pytest.raises(CapabilityError, match="memory budget"):
        _bind_review_input(plan, state)


def test_replay_never_compiles_or_replans(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _review_plan(fq.Circuit(1, dtype=torch.complex128).x(0))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("accepted review plan was recompiled or replanned")

    monkeypatch.setattr("flagquantum.runtime.planner.compile_program", forbidden)
    monkeypatch.setattr("flagquantum.runtime.planner.plan_advanced", forbidden)
    assert plan._contract_payload_json is not None
    plan = _restore_review_plan(plan._contract_payload_json)
    torch.testing.assert_close(
        _execute_review(plan, torch.tensor([0, 1], dtype=torch.complex128)),
        torch.tensor([1, 0], dtype=torch.complex128),
    )


def test_zero_plan_and_non_gradient_plan_refuse_incompatible_input() -> None:
    circuit = fq.Circuit(1, dtype=torch.complex128)
    state = torch.tensor([0, 1], dtype=torch.complex128, requires_grad=True)
    with pytest.raises(CapabilityError, match="initial-state extension"):
        _execute_review(fq.plan(circuit), state)
    with pytest.raises(CapabilityError, match="gradient-capable plan"):
        _execute_review(_review_plan(circuit), state)


def test_identity_and_extension_tampering_fail_closed() -> None:
    plan = _review_plan(fq.Circuit(1, dtype=torch.complex128))
    payload = _review_payload(plan)
    payload["identity"] = "tampered"
    with pytest.raises(ValidationError, match="identity mismatch"):
        _review_payload(
            replace(plan, _contract_payload_json=codec.canonical_json(payload))
        )
    payload["extensions"][0]["payload"]["representation"] = "density_matrix"
    with pytest.raises(CapabilityError, match="initial-state extension"):
        _review_payload(replace(plan, _contract_payload_json=_seal(payload)))


@pytest.mark.parametrize(
    "mode", ["density_matrix", "mps", "tensor_network", "stabilizer"]
)
def test_other_representations_remain_explicitly_unsupported(mode: str) -> None:
    with pytest.raises(CapabilityError, match="exact CPU statevector"):
        _review_plan(fq.Circuit(1), options=fq.ExecutionOptions(mode=mode))


def test_reset_is_rejected_before_zero_state_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = fq.Circuit.from_ir(
        CircuitIR(1, (Instruction("reset", (0,), metadata={"is_dynamic": True}),))
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unsupported reset reached compiler")

    monkeypatch.setattr("flagquantum.runtime.planner.compile_program", forbidden)
    with pytest.raises(CapabilityError, match="unitary opcodes"):
        _review_plan(circuit)
