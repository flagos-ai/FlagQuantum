"""Named angle execution preserves symbolic plans and per-attempt gradients.

Load execution's bound planner aliases before any test installs planner spies.
"""

from __future__ import annotations

import copy
import gc
import weakref
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.core.noise import NoiseModel
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError, SerializationError, ValidationError
from flagquantum.runtime import execution
from flagquantum.runtime import execution_plan_contract as codec
from flagquantum.runtime._runtime_inputs import bind_runtime_inputs
from tests.unit import test_execution_runtime_inputs_contract_fake as review

pytestmark = pytest.mark.integration


def _program(dtype: torch.dtype = torch.complex128) -> fq.Circuit:
    theta = fq.Parameter("theta")
    return fq.Circuit(1, dtype=dtype).ry(0, 2 * theta + 0.1)


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError(
        "accepted plan reached planner, compiler or numerical execution"
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_public_plan_replay_keeps_multiple_graphs_independent(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    source = _program(dtype)
    angle = torch.tensor(0.37, dtype=real, requires_grad=True)
    first = fq.run(source, inputs={"theta": angle}, outputs=fq.probabilities())
    plan = first.plan
    assert isinstance(plan, fq.ExecutionPlan)
    assert plan.to_dict()["extensions"] == [
        codec.runtime_inputs_extension(source.to_ir())
    ]
    assert plan.to_dict()["decision"]["require_gradients"] is True
    torch.testing.assert_close(first.probabilities, first.statevector().abs().square())
    snapshot = plan.to_json()
    restored = fq.ExecutionPlan.from_json(snapshot)
    assert restored.identity == plan.identity
    monkeypatch.setattr("flagquantum.runtime.planner.plan", _forbidden)
    monkeypatch.setattr("flagquantum.runtime.planner.plan_advanced", _forbidden)
    monkeypatch.setattr(execution, "build_plan", _forbidden)
    monkeypatch.setattr("flagquantum.runtime.planner.compile_program", _forbidden)
    monkeypatch.setattr("flagquantum.runtime.execution.compile_program", _forbidden)
    for candidate in (plan, restored):
        angles = [
            torch.tensor(x, dtype=real, requires_grad=True) for x in (0.0, -0.5, 0.9)
        ]
        results = [fq.run(candidate, inputs={"theta": value}) for value in angles]
        for value, result in zip(angles, results, strict=True):
            expected = torch.stack(((value + 0.05).cos(), (value + 0.05).sin())).to(
                dtype
            )
            torch.testing.assert_close(result.statevector()[0], expected)
            torch.testing.assert_close(
                torch.autograd.grad(result.statevector()[0, 1].abs().square(), value)[
                    0
                ],
                (2 * value + 0.1).sin(),
            )
            assert result.plan is candidate
        assert candidate.to_json() == snapshot
        with pytest.raises(ValidationError, match="requires inputs"):
            fq.run(candidate)
    torch.testing.assert_close(
        torch.autograd.grad(first.statevector()[0, 1].abs().square(), angle)[0],
        (2 * angle + 0.1).sin(),
    )
    assert source.parameter_names == ("theta",)


def _review_bind(plan: fq.ExecutionPlan, inputs: object) -> fq.CircuitIR:
    values = review._checked_inputs(plan, inputs)
    return review._bind_core(codec.plan_program(plan), values)


@pytest.mark.parametrize(
    "binder", [_review_bind, bind_runtime_inputs], ids=["fake", "runtime"]
)
def test_same_numerical_consumer_with_replaced_binder(
    binder: Callable[[fq.ExecutionPlan, object], fq.CircuitIR],
) -> None:
    plan = review._review_plan(_program(), gradients=True)
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    program = binder(plan, {"theta": angle})
    state = fq.Circuit.from_ir(program).state()
    assert isinstance(state, torch.Tensor)
    expected = torch.stack(((angle + 0.05).cos(), (angle + 0.05).sin())).to(
        torch.complex128
    )
    torch.testing.assert_close(state[0], expected)
    torch.testing.assert_close(
        torch.autograd.grad(state[0, 1].abs().square(), angle)[0],
        (2 * angle + 0.1).sin(),
    )
    bound = program.instructions[0].params["theta"]
    assert isinstance(bound, torch.Tensor) and bound.grad_fn is not None
    assert isinstance(
        codec.plan_program(plan).instructions[0].params["theta"], fq.ParameterExpression
    )


def test_optimizer_updates_preserve_caller_tensor_identity() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float64))
    source = fq.Circuit(1, dtype=torch.complex128).ry(0, fq.Parameter("theta"))
    first = fq.run(source, inputs={"theta": parameter})
    assert isinstance(first.plan, fq.ExecutionPlan)
    plan = first.plan
    del first
    bound = bind_runtime_inputs(plan, {"theta": parameter})
    assert bound.instructions[0].params["theta"] is parameter
    optimizer = torch.optim.SGD([parameter], lr=0.2)
    losses = []
    identity = plan.identity
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        result = fq.run(plan, inputs={"theta": parameter})
        loss = result.statevector()[0, 1].abs().square()
        losses.append(float(loss.detach()))
        torch.autograd.backward(loss)
        assert parameter.grad is not None
        optimizer.step()
    assert losses[-1] < losses[0] and plan.identity == identity


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_entangled_execution_matches_independent_reference(dtype: torch.dtype) -> None:
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    a, b = fq.Parameter("a"), fq.Parameter("b")
    source = fq.Circuit(2, dtype=dtype).ry(0, a).cx(0, 1).rz(1, b)
    left = torch.tensor(0.6, dtype=real, requires_grad=True)
    right = torch.tensor(-0.3, dtype=real, requires_grad=True)
    result = fq.run(source, inputs={"b": right, "a": left})
    zero = torch.zeros((), dtype=dtype)
    expected = torch.stack(
        (
            (left / 2).cos() * torch.exp(-0.5j * right),
            zero,
            zero,
            (left / 2).sin() * torch.exp(0.5j * right),
        )
    )
    torch.testing.assert_close(result.statevector()[0], expected)
    torch.testing.assert_close(
        torch.autograd.grad(result.statevector().real.sum(), (left, right)),
        torch.autograd.grad(expected.real.sum(), (left, right)),
    )


@pytest.mark.parametrize(
    "inputs,error",
    [
        ({}, ValidationError),
        ({"wrong": 0.2}, ValidationError),
        ({"theta": 0.2, "extra": 0.4}, ValidationError),
        ({1: 0.2}, TypeError),
        ([], TypeError),
        ({"theta": True}, TypeError),
        ({"theta": 1}, TypeError),
        ({"theta": 0.2j}, TypeError),
        ({"theta": float("nan")}, ValidationError),
        ({"theta": torch.tensor(0.2, dtype=torch.float32)}, ValidationError),
        ({"theta": torch.tensor([0.2], dtype=torch.float64)}, ValidationError),
        ({"theta": torch.tensor(0.2j, dtype=torch.complex128)}, ValidationError),
        ({"theta": torch.tensor(float("inf"), dtype=torch.float64)}, ValidationError),
        ({"theta": torch.tensor(0.2, device="meta")}, CapabilityError),
    ],
)
def test_invalid_inputs_fail_before_execution(
    inputs: Any, error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _program()
    result = fq.run(source, inputs={"theta": 0.2})
    monkeypatch.setattr("flagquantum.runtime.execution.run_native", _forbidden)
    for candidate in (source, result.plan):
        with pytest.raises(error):
            fq.run(candidate, inputs=inputs)


def test_expression_and_float_conversion_overflow_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _program()
    plan = fq.run(source, inputs={"theta": 0.2}).plan
    monkeypatch.setattr("flagquantum.runtime.execution.run_native", _forbidden)
    for candidate in (source, plan):
        with pytest.raises(ValidationError, match="finite"):
            fq.run(candidate, inputs={"theta": 1e308})
    with pytest.raises(ValidationError, match="finite"):
        fq.run(_program(torch.complex64), inputs={"theta": 1e300})


def test_gradient_policy_and_precision_promotion() -> None:
    value = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValidationError, match="require_gradients=False"):
        fq.run(
            _program(),
            inputs={"theta": value},
            options=fq.ExecutionOptions(require_gradients=False),
        )
    plan = fq.run(_program(), inputs={"theta": 0.2}).plan
    with pytest.raises(CapabilityError, match="gradient-capable"):
        fq.run(plan, inputs={"theta": value})
    result = fq.run(
        _program(torch.complex64),
        inputs={"theta": value},
        options=fq.ExecutionOptions(precision="complex128"),
    )
    assert result.statevector().dtype == torch.complex128
    assert result.plan.to_dict()["decision"]["require_gradients"] is True
    torch.autograd.backward(result.statevector().real.sum())
    assert value.grad is not None


def test_unmarked_and_conflicting_plans_fail_before_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _program()
    plan = fq.plan(source)
    named_plan = fq.run(source, inputs={"theta": 0.2}).plan
    initial_plan = fq.run(
        fq.Circuit(1, dtype=torch.complex128),
        initial_state=torch.tensor([1, 0], dtype=torch.complex128),
    ).plan
    monkeypatch.setattr("flagquantum.runtime.planner.plan", _forbidden)
    monkeypatch.setattr("flagquantum.runtime.planner.plan_advanced", _forbidden)
    monkeypatch.setattr(execution, "build_plan", _forbidden)
    for candidate in (plan, initial_plan):
        with pytest.raises(CapabilityError, match="does not accept runtime inputs"):
            fq.run(candidate, inputs={"theta": 0.2})
    for candidate in (source, named_plan):
        with pytest.raises(CapabilityError, match="cannot be combined"):
            fq.run(candidate, inputs={"theta": 0.2}, initial_state=torch.tensor([1, 0]))
    with pytest.raises(CapabilityError, match="cannot be combined"):
        fq.run(named_plan, initial_state=torch.tensor([1, 0]))


@pytest.mark.parametrize(
    "options",
    [
        fq.ExecutionOptions(mode="density_matrix"),
        fq.ExecutionOptions(mode="mps"),
        fq.ExecutionOptions(mode="tensor_network"),
        fq.ExecutionOptions(mode="stabilizer"),
        fq.ExecutionOptions(backend="jax"),
        fq.ExecutionOptions(device="cuda"),
        fq.ExecutionOptions(batch_size=2),
        fq.ExecutionOptions(shots=10),
        fq.ExecutionOptions(allow_approximate=True),
        fq.ExecutionOptions(allow_backend_fallback=True),
    ],
)
def test_unsupported_routes_fail_before_planner(
    options: fq.ExecutionOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flagquantum.runtime.planner.plan_advanced", _forbidden)
    monkeypatch.setattr(execution, "build_plan", _forbidden)
    expected = ValueError if options.batch_size == 2 else CapabilityError
    with pytest.raises(expected):
        fq.run(_program(), inputs={"theta": 0.2}, options=options)


def test_unsupported_sources_and_tensor_capture_fail_closed() -> None:
    theta = fq.Parameter("theta")
    for source in (
        _program().to_ir(),
        DynamicCircuit(1).ry(0, theta).measure(0),
        fq.Circuit(1, bsz=2).ry(0, theta),
        fq.Circuit(1, inputs=torch.tensor([1, 0])).ry(0, theta),
        fq.Circuit(1).ry(0, theta + torch.tensor(0.1)),
        fq.Circuit(1).ry(0, [theta, theta]),
        fq.Circuit(1).gate("ry", (0,), params={"theta": theta, "unused": 0.2}),
        fq.Circuit(1).ry(0, theta).any(0, matrix=torch.eye(2)),
    ):
        with pytest.raises((CapabilityError, ValidationError)):
            fq.run(source, inputs={"theta": 0.2})
    for kwargs in (
        {"shots": 1},
        {"target": "provider:device"},
        {"compiler": "flagquantum"},
        {"name": "job"},
    ):
        with pytest.raises(CapabilityError):
            fq.run(_program(), inputs={"theta": 0.2}, **kwargs)


def test_plan_and_extension_tampering_fail_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _program()
    plan = fq.run(source, inputs={"theta": 0.2}).plan
    assert isinstance(plan, fq.ExecutionPlan)
    payload = plan.to_dict()
    monkeypatch.setattr("flagquantum.runtime.execution.run_native", _forbidden)
    stale = replace(plan, layers=fq.plan(source.x(0)).layers)
    with pytest.raises(codec.ExecutionPlanContractError, match="symbolic template"):
        fq.run(stale, inputs={"theta": 0.2})
    with pytest.raises(ValidationError, match="state_bytes"):
        fq.run(replace(plan, state_bytes=0), inputs={"theta": 0.2})
    for change in ("names", "precision", "version", "mode", "tensor", "memory"):
        changed = copy.deepcopy(payload)
        if change == "names":
            changed["extensions"][0]["payload"]["parameters"] = ["other"]
            changed["extensions"][0]["identity"] = codec.canonical_hash(
                changed["extensions"][0]["payload"]
            )
        elif change == "precision":
            changed["decision"]["precision"] = "complex64"
        elif change == "version":
            changed["extensions"][0]["version"] = "unsupported"
        elif change == "mode":
            changed["decision"]["mode"] = "mps"
        elif change == "memory":
            changed["decision"]["memory_limit_bytes"] = 1
            changed["resolved_options"]["memory_limit_bytes"] = 1
        else:
            changed["extensions"][0]["payload"]["tensor"] = [0.2]
        changed["identity"] = codec._identity(
            changed, compiler=codec._compiler_identity()
        )
        with pytest.raises(codec.ExecutionPlanContractError):
            fq.ExecutionPlan.from_dict(changed)


def test_fqx_preserves_input_requirement_without_execution_on_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = fq.run(_program(), inputs={"theta": 0.2})
    plan = result.plan
    assert isinstance(plan, fq.ExecutionPlan)
    path = tmp_path / "angles.fqx"
    fq.save(path, plan, result=result)
    text = plan.to_json()
    with monkeypatch.context() as context:
        context.setattr("flagquantum.runtime.execution.run_native", _forbidden)
        loaded = fq.load(path)
    assert loaded.result is not None and loaded.result.plan is None
    assert loaded.execution_plan is not None
    restored = fq.ExecutionPlan.from_dict(loaded.execution_plan)
    assert restored.to_json() == text
    replay = fq.run(restored, inputs={"theta": 0.8})
    torch.testing.assert_close(
        replay.statevector(), fq.run(_program(), inputs={"theta": 0.8}).statevector()
    )
    refused = tmp_path / "requires_inputs.fqx"
    with pytest.raises(SerializationError, match="requires its result"):
        fq.save(refused, plan)
    assert not refused.exists()


def test_ordinary_no_input_workflow_is_unchanged() -> None:
    source = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    direct = fq.run(source)
    plan = fq.plan(source)
    replay = fq.run(plan)
    assert direct.plan.to_json() == plan.to_json()
    assert plan.to_dict()["extensions"] == []
    torch.testing.assert_close(direct.statevector(), replay.statevector())


def test_plan_does_not_retain_input_storage_or_graph() -> None:
    angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    reference = weakref.ref(angle)
    result = fq.run(_program(), inputs={"theta": angle})
    plan = result.plan
    assert isinstance(plan, fq.ExecutionPlan)
    del result, angle
    gc.collect()
    assert reference() is None
    assert review._names(codec.plan_program(plan)) == ("theta",)


def test_resource_and_environment_failures_precede_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _program()
    result = fq.run(source, inputs={"theta": 0.2})
    monkeypatch.setattr("flagquantum.runtime.execution.run_native", _forbidden)
    with monkeypatch.context() as context:
        context.setattr("flagquantum.runtime.planner.plan_advanced", _forbidden)
        with pytest.raises(CapabilityError, match="state memory budget"):
            fq.run(
                source,
                inputs={"theta": 0.2},
                options=fq.ExecutionOptions(memory_limit_bytes=31),
            )
    from flagquantum.runtime.distributed.backend_policy import (
        resolve_distributed_backend_policy,
    )

    policy = resolve_distributed_backend_policy()
    monkeypatch.setattr(
        "flagquantum.runtime.distributed.backend_policy.resolve_distributed_backend_policy",
        lambda: replace(policy, profile="development", local_world_size=2),
    )
    with pytest.raises(CapabilityError, match="unbatched CPU"):
        fq.run(source, inputs={"theta": 0.2})
    with pytest.raises(codec.ExecutionPlanContractError, match="world_size_mismatch"):
        fq.run(result.plan, inputs={"theta": 0.2})


def test_replay_rejects_execution_overrides() -> None:
    result = fq.run(_program(), inputs={"theta": 0.2})
    for kwargs in (
        {"options": fq.ExecutionOptions()},
        {"outputs": fq.probabilities()},
    ):
        with pytest.raises(TypeError, match="must be None"):
            fq.run(result.plan, inputs={"theta": 0.2}, **kwargs)


def test_tensor_subclasses_and_noise_are_not_silently_admitted() -> None:
    class Angle(torch.Tensor):
        pass

    value = torch.tensor(0.2, dtype=torch.float64).as_subclass(Angle)
    with pytest.raises(TypeError, match="Python floats or scalar tensors"):
        fq.run(_program(), inputs={"theta": value})
    with pytest.raises(CapabilityError, match="no noise"):
        fq.run(_program(), inputs={"theta": 0.2}, noise_model=NoiseModel())
