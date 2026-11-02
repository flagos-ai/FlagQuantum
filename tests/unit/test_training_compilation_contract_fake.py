"""Review-only lazy Torch compiler fake, not a FlagQuantum implementation."""

import copy

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch._dynamo.reset()
    torch.set_num_threads(previous)


def builder(p, x):
    return fq.Circuit(1, bsz=x.shape[0], dtype=torch.complex128).ry(
        0, p[0].square() + p[1] * x[:, 0].sin()
    )


def model_and_optimizer(kind):
    model = fq.Module(
        builder,
        2,
        dtype=torch.float64,
        init=torch.tensor([0.2, -0.3], dtype=torch.float64),
    )
    return model, getattr(torch.optim, kind)(model.parameters(), lr=0.03)


def record_forward(events):
    def hook(*args):
        events.append("forward")

    return hook


def numerical_step(model, optimizer):
    def step(x):
        optimizer.zero_grad(set_to_none=True)
        value = model(x)
        weights = torch.linspace(-0.7, 1.3, x.shape[0], dtype=x.dtype)
        loss = (value * weights).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    return step


def replace(step, model, kind, use_fake):
    receipt = {"route": "eager", "graphs": [], "reason": "not_requested"}
    if not use_fake:
        return step, receipt
    if kind != "SGD":
        receipt["reason"] = "adam_compiled_float64_trajectory_not_conformant"
        return step, receipt
    if model._forward_hooks:
        receipt["reason"] = "user_hooks_require_eager"
        return step, receipt

    def identity(graph, examples):
        receipt["graphs"].append(tuple(str(node.target) for node in graph.graph.nodes))
        return graph.forward

    receipt.update(
        route="identity_compiler_fake", reason="graph_breaks_allowed_not_whole_graph"
    )
    return torch.compile(step, backend=identity, fullgraph=False), receipt


@pytest.mark.parametrize("batch", (1, 8))
@pytest.mark.parametrize("kind", ("SGD", "Adam"))
@pytest.mark.parametrize("hooks", (False, True))
def test_same_public_consumer_and_recovery(batch, kind, hooks, tmp_path):
    trajectories = []
    for use_fake in (False, True):
        torch._dynamo.reset()
        model, optimizer = model_and_optimizer(kind)
        reference = model.parameters_tensor.detach().clone().requires_grad_()
        reference_optimizer = getattr(torch.optim, kind)([reference], lr=0.03)
        events = []
        if hooks:
            model.register_forward_hook(record_forward(events))
        step, receipt = replace(numerical_step(model, optimizer), model, kind, use_fake)
        losses = []
        gradients = []
        restored = None
        for index in range(6):
            x = (
                torch.linspace(-0.3, 0.5, batch, dtype=torch.float64)[:, None]
                + index * 0.01
            ).requires_grad_()
            reference_x = x.detach().clone().requires_grad_()
            reference_optimizer.zero_grad(set_to_none=True)
            expected = (
                reference[0].square() + reference[1] * reference_x[:, 0].sin()
            ).cos()
            weights = torch.linspace(-0.7, 1.3, batch, dtype=x.dtype)
            expected_loss = (expected * weights).square().mean()
            expected_loss.backward()
            losses.append(float(step(x)))
            torch.testing.assert_close(
                losses[-1], float(expected_loss.detach()), atol=1e-12, rtol=1e-12
            )
            torch.testing.assert_close(
                model.parameters_tensor.grad, reference.grad, atol=1e-12, rtol=1e-12
            )
            torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-12, rtol=1e-12)
            reference_optimizer.step()
            torch.testing.assert_close(
                model.parameters_tensor, reference, atol=1e-12, rtol=1e-12
            )
            gradients.append(
                (model.parameters_tensor.grad.detach().clone(), x.grad.detach().clone())
            )
            if restored is not None:
                replay, replay_optimizer, replay_step = restored
                replay_x = x.detach().clone().requires_grad_()
                replay_step(replay_x)
                torch.testing.assert_close(
                    replay.parameters_tensor,
                    model.parameters_tensor,
                    atol=1e-12,
                    rtol=1e-12,
                )
                assert replay._last_ir.content_hash == model._last_ir.content_hash
            if index == 2:
                checkpoint = model.save_checkpoint(
                    tmp_path / f"{use_fake}.pt", optimizer=optimizer, seed=11, step=3
                )
                replay, replay_optimizer = model_and_optimizer(kind)
                replay(x.detach())
                replay.load_checkpoint(checkpoint, optimizer=replay_optimizer)
                replay_step, _ = replace(
                    numerical_step(replay, replay_optimizer), replay, kind, use_fake
                )
                restored = replay, replay_optimizer, replay_step
        with torch.no_grad():
            assert torch.isfinite(model(x)).all()
        trajectories.append(
            (
                losses,
                model.parameters_tensor.detach().clone(),
                gradients,
                copy.deepcopy(optimizer.state_dict()),
                events,
            )
        )
        if use_fake and kind == "SGD" and not hooks:
            assert receipt["route"] == "identity_compiler_fake" and receipt["graphs"]
        elif use_fake:
            assert receipt["route"] == "eager" and receipt["reason"] != "not_requested"
    for eager, replacement in zip(
        trajectories[0][:3], trajectories[1][:3], strict=True
    ):
        torch.testing.assert_close(eager, replacement, atol=1e-12, rtol=1e-12)
    assert trajectories[0][4] == trajectories[1][4]
    assert trajectories[0][3]["param_groups"] == trajectories[1][3]["param_groups"]
    for key, state in trajectories[0][3]["state"].items():
        torch.testing.assert_close(
            state, trajectories[1][3]["state"][key], atol=1e-12, rtol=1e-12
        )


def test_existing_train_consumer_with_identity_compiler():
    results = []
    graphs = []

    def identity(graph, examples):
        graphs.append(graph)
        return graph.forward

    for compiled in (False, True):
        torch._dynamo.reset()
        model, optimizer = model_and_optimizer("SGD")
        inputs = torch.linspace(-0.3, 0.5, 8, dtype=torch.float64)[
            :, None
        ].requires_grad_()
        invoke = (
            torch.compile(fq.train, backend=identity, fullgraph=False)
            if compiled
            else fq.train
        )
        result = invoke(
            model,
            optimizer=optimizer,
            objective=lambda value: value.square().mean(),
            inputs=inputs,
            steps=6,
        )
        results.append(
            (
                result.losses,
                model.parameters_tensor.detach().clone(),
                inputs.grad.detach().clone(),
                result.last_execution.provenance["ir_hash"],
            )
        )
    torch.testing.assert_close(results[0][:3], results[1][:3], atol=1e-12, rtol=1e-12)
    assert results[0][3] == results[1][3]
    assert graphs
