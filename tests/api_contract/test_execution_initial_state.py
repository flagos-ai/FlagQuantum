"""External state is execution data; plans seal requirements, never amplitudes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError, SerializationError, ValidationError
from flagquantum.runtime._initial_state import bind_initial_state
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
    _compiler_identity,
    _identity,
    initial_state_extension,
)
from tests.unit import test_execution_initial_state_contract_fake as review

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_public_execution_reuses_plan_for_different_complex_inputs(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    circuit = fq.Circuit(2, dtype=dtype).cx(0, 1).s(0)
    initial = torch.tensor([1, 0, 0, 1j], dtype=dtype) / 2**0.5
    result = fq.run(circuit, initial_state=initial, outputs=fq.probabilities())
    plan = result.plan
    assert isinstance(plan, fq.ExecutionPlan)
    assert plan.to_dict()["extensions"] == [initial_state_extension()]
    expected = torch.tensor([[1, 0, -1, 0]], dtype=dtype) / 2**0.5
    torch.testing.assert_close(result.statevector(), expected)
    torch.testing.assert_close(result.probabilities, expected.abs().square())
    text = plan.to_json()
    restored = fq.ExecutionPlan.from_json(text)
    assert restored.identity == plan.identity and restored.noisy_execution_plan is None

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("accepted input plan reached compiler or planner")

    monkeypatch.setattr("flagquantum.runtime.planner.compile_program", forbidden)
    monkeypatch.setattr("flagquantum.runtime.planner.plan", forbidden)
    monkeypatch.setattr("flagquantum.runtime.planner.plan_advanced", forbidden)
    next_state = torch.tensor([0, 0, 0, 1], dtype=dtype)
    for candidate in (plan, restored):
        replay = fq.run(candidate, initial_state=next_state)
        torch.testing.assert_close(
            replay.statevector(), torch.tensor([[0, 0, 1j, 0]], dtype=dtype)
        )
        assert replay.plan is candidate and candidate.to_json() == text
        with pytest.raises(ValidationError, match="requires initial_state"):
            fq.run(candidate)
    torch.testing.assert_close(
        initial, torch.tensor([1, 0, 0, 1j], dtype=dtype) / 2**0.5
    )


@pytest.mark.parametrize("replacement", [False, True], ids=["review-fake", "runtime"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_same_consumer_with_replaced_input_binding(
    replacement: bool, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    if replacement:
        monkeypatch.setattr(review, "_bind_review_input", bind_initial_state)
    # The numerical consumer and its callers are identical for both binders.
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    state = torch.nn.Parameter(torch.tensor([0.6, 99, 0.8j, 99], dtype=dtype)[::2])
    theta = torch.tensor(0.37, dtype=real_dtype, requires_grad=True)
    plan = review._review_plan(
        fq.Circuit(1, dtype=dtype).ry(0, theta), requires_grad=True
    )
    output = review._execute_review(plan, state)
    cosine, sine = (theta / 2).cos(), (theta / 2).sin()
    dense = torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2).to(dtype)
    expected = dense @ state
    torch.testing.assert_close(output, expected)
    objective = output[0].abs().square() + 0.3 * output[1].real
    reference = expected[0].abs().square() + 0.3 * expected[1].real
    torch.testing.assert_close(
        torch.autograd.grad(objective, (state, theta)),
        torch.autograd.grad(reference, (state, theta)),
    )
    owned = review._bind_review_input(plan, state)
    assert owned.is_contiguous() and owned.data_ptr() != state.data_ptr()
    with torch.no_grad():
        state.zero_()
    torch.testing.assert_close(owned, torch.tensor([0.6, 0.8j], dtype=dtype))


def test_public_input_and_gate_optimizer_updates_and_json_input_gradient() -> None:
    raw = torch.nn.Parameter(torch.tensor([0.6, 0.8j], dtype=torch.complex128))
    theta = torch.nn.Parameter(torch.tensor(0.37, dtype=torch.float64))
    optimizer = torch.optim.SGD([raw, theta], lr=0.05)
    losses = []
    plan = None
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        initial = raw / torch.linalg.vector_norm(raw)
        result = fq.run(
            fq.Circuit(1, dtype=torch.complex128).ry(0, theta),
            initial_state=initial,
        )
        plan = result.plan
        assert isinstance(plan, fq.ExecutionPlan)
        assert plan.to_dict()["decision"]["require_gradients"] is True
        loss = result.statevector()[0, 1].abs().square()
        losses.append(float(loss.detach()))
        torch.autograd.backward(loss)
        assert raw.grad is not None and theta.grad is not None
        optimizer.step()
    assert losses[-1] < losses[0]
    assert isinstance(plan, fq.ExecutionPlan)
    restored = fq.ExecutionPlan.from_json(plan.to_json())
    fresh = torch.tensor([0.6, 0.8j], dtype=torch.complex128, requires_grad=True)
    result = fq.run(restored, initial_state=fresh)
    torch.autograd.backward(result.statevector().real.sum())
    assert fresh.grad is not None


def test_changed_gate_bindings_invalidate_replay_before_input_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = torch.tensor([0, 1], dtype=torch.complex128)
    theta = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta)
    original = fq.run(circuit, initial_state=state)
    plan = original.plan
    assert isinstance(plan, fq.ExecutionPlan)
    snapshot = plan.to_json()
    with torch.no_grad():
        theta.add_(0.2)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("stale plan reached copying or numerical execution")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "clone", forbidden)
        context.setattr("flagquantum.runtime.execution.run_native", forbidden)
        with pytest.raises(ExecutionPlanContractError, match="gate bindings changed"):
            fq.run(plan, initial_state=state)
    restored = fq.ExecutionPlan.from_json(snapshot)
    torch.testing.assert_close(
        fq.run(restored, initial_state=state).statevector(), original.statevector()
    )
    updated = fq.run(circuit, initial_state=state)
    assert updated.plan.identity != plan.identity


def test_input_and_parameter_gradient_inference_and_explicit_conflicts() -> None:
    initial = torch.tensor([0, 1], dtype=torch.complex128, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128)
    with pytest.raises(ValidationError, match="require_gradients=False"):
        fq.run(
            circuit,
            initial_state=initial,
            options=fq.ExecutionOptions(require_gradients=False),
        )
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValidationError, match="require_gradients=False"):
        fq.run(
            circuit.ry(0, theta),
            initial_state=initial.detach(),
            options=fq.ExecutionOptions(require_gradients=False),
        )
    plain = fq.run(
        fq.Circuit(1, dtype=torch.complex128), initial_state=initial.detach()
    ).plan
    with pytest.raises(CapabilityError, match="gradient-capable"):
        fq.run(plain, initial_state=initial)
    zero = fq.plan(fq.Circuit(1, dtype=torch.complex128))
    with pytest.raises(CapabilityError, match="does not accept"):
        fq.run(zero, initial_state=initial)


def test_resolved_precision_controls_input_admission_without_coercion() -> None:
    initial = torch.tensor([0, 1], dtype=torch.complex128)
    circuit = fq.Circuit(1, dtype=torch.complex64).x(0)
    with pytest.raises(ValidationError, match="dtype"):
        fq.run(circuit, initial_state=initial)
    result = fq.run(
        circuit,
        initial_state=initial,
        options=fq.ExecutionOptions(precision="complex128"),
    )
    assert result.statevector().dtype == torch.complex128
    assert result.plan.precision == "complex128"
    torch.testing.assert_close(
        result.statevector(), torch.tensor([[1, 0]], dtype=torch.complex128)
    )


def test_tensor_subclasses_and_distributed_requests_fail_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TensorSubclass(torch.Tensor):
        pass

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unsupported request reached compilation")

    initial = torch.tensor([0, 1], dtype=torch.complex64)
    monkeypatch.setattr("flagquantum.runtime.planner.compile_program", forbidden)
    with pytest.raises(CapabilityError, match="subclasses"):
        fq.run(fq.Circuit(1), initial_state=initial.as_subclass(TensorSubclass))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with pytest.raises(CapabilityError, match="CPU PyTorch statevector"):
        fq.run(fq.Circuit(1), initial_state=initial)


@pytest.mark.parametrize(
    "state, error",
    [
        ([1, 0], TypeError),
        (torch.ones(1, 2, dtype=torch.complex128), ValidationError),
        (torch.ones(4, dtype=torch.complex128), ValidationError),
        (torch.tensor([1, 0], dtype=torch.complex64), ValidationError),
        (torch.tensor([1, 0], dtype=torch.float64), ValidationError),
        (torch.zeros(2, dtype=torch.complex128), ValidationError),
        (torch.tensor([float("nan"), 0], dtype=torch.complex128), ValidationError),
        (torch.tensor([float("inf"), 0], dtype=torch.complex128), ValidationError),
        (torch.eye(2, dtype=torch.complex128).to_sparse(), CapabilityError),
        (torch.empty(2, dtype=torch.complex128, device="meta"), CapabilityError),
    ],
)
def test_bad_inputs_never_reach_numerical_execution(
    state: Any, error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid input reached numerical execution")

    monkeypatch.setattr("flagquantum.runtime.execution.run_native", forbidden)
    with pytest.raises(error):
        fq.run(fq.Circuit(1, dtype=torch.complex128), initial_state=state)


def test_validation_reads_later_chunks_before_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = torch.zeros(8192, dtype=torch.complex128)
    state[-1] = torch.tensor(1j, dtype=state.dtype)
    circuit = fq.Circuit(13, dtype=torch.complex128).x(0)
    result = fq.run(circuit, initial_state=state).statevector().reshape(-1)
    assert result[4095] == 1j and int(torch.count_nonzero(result)) == 1
    state[-1] = torch.tensor(complex(float("nan"), 0), dtype=state.dtype)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("nonfinite input reached copying")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    with pytest.raises(ValidationError, match="finite"):
        fq.run(circuit, initial_state=state)


@pytest.mark.parametrize("limit", [1, 32, 63, 64])
def test_memory_minimum_precedes_compilation_and_clone(
    limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = torch.tensor([0, 1], dtype=torch.complex128)
    if limit < 64:

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("insufficient budget reached compilation or copying")

        monkeypatch.setattr("flagquantum.runtime.planner.compile_program", forbidden)
        monkeypatch.setattr(torch.Tensor, "clone", forbidden)
        with pytest.raises(CapabilityError, match="memory budget"):
            fq.run(
                fq.Circuit(1, dtype=torch.complex128),
                initial_state=initial,
                options=fq.ExecutionOptions(memory_limit_bytes=limit),
            )
    else:
        result = fq.run(
            fq.Circuit(1, dtype=torch.complex128),
            initial_state=initial,
            options=fq.ExecutionOptions(memory_limit_bytes=limit),
        )
        torch.testing.assert_close(result.statevector().reshape(-1), initial)


@pytest.mark.parametrize(
    "mode", ["density_matrix", "mps", "tensor_network", "stabilizer"]
)
def test_other_engines_fail_before_compilation(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unsupported engine reached compilation")

    monkeypatch.setattr("flagquantum.runtime.planner.compile_program", forbidden)
    with pytest.raises(CapabilityError, match="CPU PyTorch statevector"):
        fq.run(
            fq.Circuit(1),
            initial_state=torch.tensor([0, 1], dtype=torch.complex64),
            options=fq.ExecutionOptions(mode=mode),
        )


def test_unsupported_sources_and_outputs_do_not_change_existing_programs() -> None:
    state = torch.tensor([0, 1], dtype=torch.complex128)
    circuit = fq.Circuit(1, dtype=torch.complex128)
    for source in (
        circuit.to_ir(),
        DynamicCircuit(1, dtype=torch.complex128).reset(0),
        circuit.gate("custom", 0, matrix=torch.eye(2, dtype=torch.complex128)),
    ):
        with pytest.raises(CapabilityError):
            fq.run(source, initial_state=state)
    with pytest.raises(TypeError, match="construction inputs"):
        fq.run(fq.Circuit(1, inputs=state, dtype=torch.complex128), initial_state=state)
    for output in (fq.density_matrix(), fq.expectation(fq.Z(0))):
        with pytest.raises(CapabilityError, match="statevectors and probabilities"):
            fq.run(
                fq.Circuit(1, dtype=torch.complex128),
                initial_state=state,
                outputs=output,
            )
    with pytest.raises(CapabilityError, match="remote"):
        fq.run(
            fq.Circuit(1, dtype=torch.complex128),
            initial_state=state,
            target="quafu:unused",
        )


def test_fqx_round_trip_retains_requirement_and_never_executes_on_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = torch.tensor([0.6, 0.8j], dtype=torch.complex128, requires_grad=True)
    result = fq.run(fq.Circuit(1, dtype=torch.complex128).h(0), initial_state=state)
    plan = result.plan
    assert isinstance(plan, fq.ExecutionPlan)
    path = tmp_path / "state.fqx"
    fq.save(path, plan, result=result)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("artifact load executed or planned a computation")

    with monkeypatch.context() as context:
        context.setattr("flagquantum.runtime.execution.run_native", forbidden)
        context.setattr("flagquantum.runtime.planner.plan", forbidden)
        loaded = fq.load(path)
    assert loaded.result is not None and loaded.result.plan is None
    assert loaded.result.statevector().grad_fn is None
    torch.testing.assert_close(loaded.result.statevector(), result.statevector())
    assert loaded.execution_plan == plan.to_dict()
    replay = fq.ExecutionPlan.from_dict(loaded.execution_plan)
    with pytest.raises(ValidationError, match="requires initial_state"):
        fq.run(replay)
    torch.testing.assert_close(
        fq.run(replay, initial_state=state).statevector(), result.statevector()
    )
    fq.save(tmp_path / "copy.fqx", loaded)
    for overwrite in (False, True):
        destination = tmp_path / "plan-only.fqx"
        if overwrite:
            destination.write_bytes(b"keep-existing-file")
        with pytest.raises(
            SerializationError, match="discard the initial_state requirement"
        ):
            fq.save(destination, plan, overwrite=overwrite)
        if overwrite:
            assert destination.read_bytes() == b"keep-existing-file"
        else:
            assert not destination.exists()
    assert not list(tmp_path.glob(".fqx-*.tmp"))


@pytest.mark.parametrize(
    "change", ["mode", "device", "world_size", "state_bytes", "target"]
)
def test_rehashed_incompatible_decisions_are_rejected(change: str) -> None:
    state = torch.tensor([0, 1], dtype=torch.complex128)
    plan = fq.run(fq.Circuit(1, dtype=torch.complex128), initial_state=state).plan
    assert isinstance(plan, fq.ExecutionPlan)
    payload = plan.to_dict()
    payload["decision"][change] = {
        "mode": "density_matrix",
        "device": "cuda",
        "world_size": 2,
        "state_bytes": 1,
        "target": "samples",
    }[change]
    payload["identity"] = _identity(payload, compiler=_compiler_identity())
    with pytest.raises(ExecutionPlanContractError, match="extension_incompatible"):
        fq.ExecutionPlan.from_dict(payload)


@pytest.mark.parametrize("change", ["signature", "scope", "release", "unknown"])
def test_api_checker_refuses_unapproved_expansion(change: str) -> None:
    from tools.public_api_snapshot import (
        INITIAL_STATE_CONTRACT,
        _validate_authorized_initial_state,
    )

    contract = json.loads(INITIAL_STATE_CONTRACT.read_text())
    assert _validate_authorized_initial_state(contract) == []
    tampered = copy.deepcopy(contract)
    if change == "signature":
        tampered["run_signature"] = tampered["run_signature"].replace(
            "Tensor | None", "object"
        )
    elif change == "scope":
        tampered["scope"] = "all_engines"
    elif change == "release":
        tampered["release_authorized"] = True
    else:
        tampered["accept_density_matrix"] = True
    assert _validate_authorized_initial_state(tampered)
