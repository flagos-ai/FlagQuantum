"""Ordinary fq.train replacement preserves values, ownership and recovery."""

import importlib

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import training as compiler
from flagquantum.runtime.result import ExecutionResult
from flagquantum.runtime.result_adapters import LiveRuntimeSummary
from flagquantum.runtime.training_compilation import _compilation_diagnostics

pytestmark = pytest.mark.unit
training = importlib.import_module("flagquantum.runtime.training")


@pytest.fixture
def candidate(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch._dynamo.reset()
    monkeypatch.setattr(training, "_MIN_COMPILED_STEPS", 0)
    original = compiler.compile_training_callable

    def identity(step, **options):
        options["backend"] = lambda graph, examples: graph.forward
        return original(step, **options)

    monkeypatch.setattr(compiler, "compile_training_callable", identity)
    yield
    torch._dynamo.reset()
    torch.set_num_threads(previous)


def build(p, x):
    return fq.Circuit(1, bsz=x.shape[0], dtype=torch.complex128).ry(
        0, p[0].square() + p[1] * x[:, 0].sin()
    )


def model():
    return fq.Module(
        build,
        2,
        dtype=torch.float64,
        init=torch.tensor([0.2, -0.3], dtype=torch.float64),
    )


@pytest.mark.parametrize("kind", ("SGD", "Adam"))
@pytest.mark.parametrize("batch", (1, 8))
@pytest.mark.parametrize("mask", ((True, False), (False, True), (True, True)))
def test_public_training_matches_analytic_values_vjps_updates_and_state(
    candidate, kind, batch, mask
):
    current = model()
    current.parameters_tensor.requires_grad_(mask[0])
    optimizer = getattr(torch.optim, kind)(current.parameters(), lr=0.03)
    reference = current.parameters_tensor.detach().clone().requires_grad_(mask[0])
    reference_optimizer = getattr(torch.optim, kind)([reference], lr=0.03)
    weights = torch.linspace(-0.7, 1.3, batch, dtype=torch.float64)
    target = torch.linspace(-0.2, 0.4, batch, dtype=torch.float64)
    supplied = []
    expected_losses = []

    def objective(value):
        return ((value - target) * weights).square().mean()

    def inputs(index):
        x = (
            torch.linspace(-0.3, 0.5, batch, dtype=torch.float64)[:, None]
            + index * 0.01
        ).requires_grad_(mask[1])
        supplied.append(
            (x, current.parameters_tensor.detach().clone().requires_grad_(mask[0]))
        )
        return x

    def callback(step, loss, execution):
        x, before = supplied[-1]
        reference_x = x.detach().clone().requires_grad_(mask[1])
        reference_optimizer.zero_grad(set_to_none=True)
        angle = reference[0].square() + reference[1] * reference_x[:, 0].sin()
        value = angle.cos()
        expected_loss = objective(value)
        expected_loss.backward()
        expected_losses.append(float(expected_loss.detach()))
        torch.testing.assert_close(loss, expected_losses[-1], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            execution.value, value.detach(), atol=1e-12, rtol=1e-12
        )
        before_angle = before[0].square() + before[1] * x[:, 0].sin()
        state = torch.stack(
            ((before_angle / 2).cos(), (before_angle / 2).sin()), -1
        ).to(torch.complex128)
        torch.testing.assert_close(execution.state, state, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            current.parameters_tensor.grad, reference.grad, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-12, rtol=1e-12)
        reference_optimizer.step()
        torch.testing.assert_close(
            current.parameters_tensor, reference, atol=1e-12, rtol=1e-12
        )
        assert execution.provenance["ir_hash"] == build(before, x).to_ir().content_hash
        assert execution.metrics["input_batch_size"] == batch
        assert execution.metrics["training"] is True
        assert (
            execution.runtime["training_compilation"]["status"]
            == "reusable_tensor_regions"
        )

    result = fq.train(
        current,
        optimizer=optimizer,
        objective=objective,
        inputs=inputs,
        steps=6,
        callback=callback,
    )
    assert len(supplied) == 6 and result.completed_steps == 6
    torch.testing.assert_close(result.losses, expected_losses, atol=1e-12, rtol=1e-12)
    assert set(current.state_dict()) == {"parameters_tensor", "_extra_state"}
    assert current._builder_compile_count == 1
    assert current._training_program is not None
    diagnostics = result.last_execution.runtime["training_compilation"]
    assert diagnostics["whole_step_fused"] is False
    assert diagnostics["value_objective_vjp"]["compiled_graph_count"] > 0
    if kind == "Adam":
        assert (
            diagnostics["optimizer"]["reason"]
            == "adam_optimizer_precision_not_certified"
        )
    for key, state in reference_optimizer.state_dict()["state"].items():
        torch.testing.assert_close(
            optimizer.state_dict()["state"][key], state, atol=1e-12, rtol=1e-12
        )


def test_live_backward_evidence_survives_training_diagnostics():
    class Evidence:
        complete = False

        def summary(self):
            return {"backward_complete": self.complete}

    evidence = Evidence()
    execution = ExecutionResult(runtime=LiveRuntimeSummary(evidence))
    observed = _compilation_diagnostics(execution, {"status": "eager"})
    evidence.complete = True
    assert observed.runtime["backward_complete"] is True


def test_default_promotion_gate_keeps_short_training_eager():
    current = model()
    result = fq.train(
        current,
        optimizer=torch.optim.Adam(current.parameters(), lr=0.03),
        objective=lambda value: value.square().mean(),
        steps=2,
        inputs=torch.tensor([[0.1]], dtype=torch.float64),
    )
    assert result.last_execution.runtime["training_compilation"]["status"] == "eager"
    assert current._training_program is None


@pytest.mark.parametrize("kind", ("SGD", "Adam"))
def test_checkpoint_recovery_drops_compiled_artifacts_and_keeps_consumed_hashes(
    candidate, kind, tmp_path
):
    current = model()
    optimizer = getattr(torch.optim, kind)(current.parameters(), lr=0.03)

    def features(index):
        return torch.tensor([[0.1], [0.3]], dtype=torch.float64) + index * 0.01

    def objective(value):
        return value.square().mean()

    def callback(step, loss, execution):
        if step == 3:
            current.save_checkpoint(
                tmp_path / "step3.pt", optimizer=optimizer, seed=11, step=3
            )

    complete = fq.train(
        current,
        optimizer=optimizer,
        objective=objective,
        inputs=features,
        steps=6,
        callback=callback,
    )
    replay = model()
    replay_optimizer = getattr(torch.optim, kind)(replay.parameters(), lr=0.03)
    replay(features(2))
    replay.load_checkpoint(tmp_path / "step3.pt", optimizer=replay_optimizer)
    assert replay._training_program is None
    restored = fq.train(
        replay,
        optimizer=replay_optimizer,
        objective=objective,
        inputs=lambda index: features(index + 3),
        steps=3,
    )
    torch.testing.assert_close(
        replay.parameters_tensor, current.parameters_tensor, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        restored.losses, complete.losses[3:], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        restored.last_execution.state,
        complete.last_execution.state,
        atol=1e-12,
        rtol=1e-12,
    )
    assert (
        restored.last_execution.provenance["ir_hash"]
        == complete.last_execution.provenance["ir_hash"]
    )
    for key, state in optimizer.state_dict()["state"].items():
        torch.testing.assert_close(
            replay_optimizer.state_dict()["state"][key], state, atol=1e-12, rtol=1e-12
        )


def test_unused_adam_parameter_does_not_acquire_a_zero_gradient_or_optimizer_slot(
    candidate,
):
    def unused(p, x):
        return fq.Circuit(1, bsz=x.shape[0], dtype=torch.complex128).ry(0, x[:, 0])

    current = fq.Module(unused, 2, dtype=torch.float64)
    optimizer = torch.optim.Adam(current.parameters(), lr=0.03)
    before = current.parameters_tensor.detach().clone()
    x = torch.tensor([[0.1], [0.3]], dtype=torch.float64, requires_grad=True)
    result = fq.train(
        current,
        optimizer=optimizer,
        objective=lambda value: value.square().mean(),
        inputs=x,
        steps=6,
    )
    assert result.completed_steps == 6 and current._training_program is not None
    assert current._training_program.gradient_indices == (1,)
    assert current.parameters_tensor.grad is None and not optimizer.state
    torch.testing.assert_close(current.parameters_tensor, before, atol=0, rtol=0)
    torch.testing.assert_close(
        x.grad[:, 0], -6 * x[:, 0].sin() * x[:, 0].cos(), atol=1e-12, rtol=1e-12
    )


def test_changing_frozen_objective_data_reuses_one_program(candidate):
    current = model()
    target = torch.zeros(2, dtype=torch.float64)
    caches = []

    def objective(value):
        return (value - target).square().mean()

    def inputs(index):
        nonlocal target
        target = torch.full((2,), index * 0.01, dtype=torch.float64)
        return torch.tensor([[0.1], [0.3]], dtype=torch.float64)

    def callback(step, loss, execution):
        caches.append(id(current._training_program))

    result = fq.train(
        current,
        optimizer=torch.optim.SGD(current.parameters(), lr=0.03),
        objective=objective,
        inputs=inputs,
        steps=6,
        callback=callback,
    )
    assert len(set(caches)) == 1 and current._training_program is not None
    assert (
        result.last_execution.runtime["training_compilation"]["value_objective_vjp"][
            "captured_graph_count"
        ]
        == 1
    )


def test_new_trainable_objective_capture_invalidates_to_eager(candidate):
    current = model()
    target = torch.zeros(2, dtype=torch.float64)
    captures = []

    def objective(value):
        return (value - target).square().mean()

    def inputs(index):
        nonlocal target
        target = torch.full(
            (2,), index * 0.01, dtype=torch.float64, requires_grad=index >= 2
        )
        captures.append(target)
        return torch.tensor([[0.1], [0.3]], dtype=torch.float64)

    result = fq.train(
        current,
        optimizer=torch.optim.SGD(current.parameters(), lr=0.03),
        objective=objective,
        inputs=inputs,
        steps=6,
    )
    assert all(value.grad is not None for value in captures[2:])
    assert current._training_program is None
    assert (
        result.last_execution.runtime["training_compilation"]["reason"]
        == "training_signature_changed"
    )


def test_stateful_objective_and_upstream_features_keep_the_original_eager_lifecycle(
    candidate,
):
    current = model()
    calls = []

    def objective(value):
        calls.append(1)
        return value.square().mean()

    result = fq.train(
        current,
        optimizer=torch.optim.Adam(current.parameters(), lr=0.03),
        objective=objective,
        inputs=torch.tensor([[0.1]], dtype=torch.float64),
        steps=4,
    )
    assert len(calls) == 4 and current._training_program is None
    assert result.last_execution.runtime["training_compilation"]["reason"].startswith(
        "objective_"
    )
    source = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    result = fq.train(
        current,
        optimizer=torch.optim.SGD(current.parameters(), lr=0.03),
        objective=lambda value: value.square().mean(),
        inputs=lambda index: source.sin(),
        steps=3,
    )
    assert source.grad is not None
    assert (
        result.last_execution.runtime["training_compilation"]["reason"]
        == "inference_or_upstream_autograd_requires_eager"
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_vector_readout_noncontiguous_features_and_eager_sgd_options(candidate, dtype):
    def vector_build(p, x):
        precision = torch.complex128 if dtype == torch.float64 else torch.complex64
        return (
            fq.Circuit(2, bsz=x.shape[0], dtype=precision)
            .ry(0, p[0].square() + p[1] * x[:, 0].sin())
            .rx(1, p[0].detach() * x[:, 1])
            .cx(0, 1)
        )

    current = fq.Module(
        vector_build,
        2,
        dtype=dtype,
        init=torch.tensor([0.2, -0.3], dtype=dtype),
        policy=fq.RuntimePolicy(observable_qubits=(0, 1)),
    )
    optimizer = torch.optim.SGD(
        current.parameters(), lr=0.03, momentum=0.5, weight_decay=0.1
    )
    reference = current.parameters_tensor.detach().clone().requires_grad_()
    reference_optimizer = torch.optim.SGD(
        [reference], lr=0.03, momentum=0.5, weight_decay=0.1
    )
    weights = torch.tensor([0.4, -0.7], dtype=dtype)

    def objective(value):
        return (value * weights).square().mean()

    def inputs(index):
        return (
            torch.linspace(-0.3, 0.5, 16, dtype=dtype).reshape(4, 4) + index * 0.01
        )[:, ::2].requires_grad_()

    observed = []

    def callback(step, loss, execution):
        x = inputs(step - 1)
        reference_optimizer.zero_grad(set_to_none=True)
        angle = reference[0].square() + reference[1] * x[:, 0].sin()
        values = torch.stack(
            (angle.cos(), angle.cos() * (reference[0].detach() * x[:, 1]).cos()), -1
        )
        expected_loss = objective(values)
        expected_loss.backward()
        reference_optimizer.step()
        tolerance = 2e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(
            loss, float(expected_loss.detach()), atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            current.parameters_tensor, reference, atol=tolerance, rtol=tolerance
        )
        observed.append(execution.runtime["training_compilation"])

    fq.train(
        current,
        optimizer=optimizer,
        objective=objective,
        inputs=inputs,
        steps=6,
        callback=callback,
    )
    assert observed[-1]["optimizer"]["reason"] == "sgd_options_not_certified"
    torch.testing.assert_close(optimizer.state_dict(), reference_optimizer.state_dict())


def test_parameter_hooks_added_during_callback_select_eager_before_next_update(
    candidate,
):
    current = model()
    gradients = []
    hooks = []

    def callback(step, loss, execution):
        if step == 1:
            hooks.append(
                current.parameters_tensor.register_hook(
                    lambda grad: gradients.append(grad.clone())
                )
            )

    result = fq.train(
        current,
        optimizer=torch.optim.SGD(current.parameters(), lr=0.03),
        objective=lambda value: value.square().mean(),
        inputs=torch.tensor([[0.1]], dtype=torch.float64),
        steps=4,
        callback=callback,
    )
    for handle in hooks:
        handle.remove()
    assert len(gradients) == 3 and current._training_program is None
    assert (
        result.last_execution.runtime["training_compilation"]["reason"]
        == "tensor_hooks_require_eager"
    )
