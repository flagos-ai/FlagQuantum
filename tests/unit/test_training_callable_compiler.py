"""Lazy compiler replacement preserves mutation count and failure semantics."""

import pytest
import torch

from flagquantum.compiler.training import compile_training_callable

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_compiler():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_compilation_is_lazy_and_does_not_replay_optimizer_updates():
    parameter = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.03)
    graphs = []
    calls = []
    diagnostics = {}

    def backend(graph, examples):
        graphs.append(graph)
        return graph.forward

    def step(x):
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter + x).square().sum()
        loss.backward()
        optimizer.step()
        calls.append(1)
        return loss.detach()

    compiled = compile_training_callable(step, backend=backend, diagnostics=diagnostics)
    assert not graphs and not calls
    reference = parameter.detach().clone()
    for index in range(3):
        value = torch.tensor([index * 0.1], dtype=torch.float64)
        expected_loss = (reference + value).square().sum()
        reference -= 0.03 * 2 * (reference + value)
        torch.testing.assert_close(compiled(value), expected_loss)
        torch.testing.assert_close(parameter, reference, atol=1e-12, rtol=1e-12)
    assert len(calls) == 3 and graphs
    assert diagnostics["compiled_graph_count"] == len(graphs)
    assert diagnostics["declined_graph_count"] == 0
    assert diagnostics["whole_step_fused"] is False


def test_backend_failure_declines_only_its_graph_without_retrying_the_step():
    value = torch.tensor([0.2], dtype=torch.float64)
    calls = []
    diagnostics = {}

    def unavailable(graph, examples):
        raise RuntimeError("compiler unavailable")

    def step(x):
        calls.append(1)
        value.add_(x)
        return value.clone()

    compiled = compile_training_callable(
        step, backend=unavailable, diagnostics=diagnostics
    )
    for _ in range(3):
        compiled(torch.tensor([0.1], dtype=torch.float64))
    torch.testing.assert_close(value, torch.tensor([0.5], dtype=torch.float64))
    assert len(calls) == 3
    assert diagnostics["compiled_graph_count"] == 0
    assert diagnostics["declined_graph_count"] > 0
    assert diagnostics["reason"] == "graph_compiler_declined:RuntimeError"


@pytest.mark.parametrize("missing", ("function", "backend"))
def test_unavailable_compilation_retains_the_original_callable(monkeypatch, missing):
    def step(value):
        return value.sin()

    if missing == "function":
        monkeypatch.setattr(torch, "compile", None)
    diagnostics = {}
    result = compile_training_callable(
        step, backend="nonexistent_fq_test_compiler", diagnostics=diagnostics
    )
    assert result is step
    assert diagnostics["status"] == "eager" and diagnostics["reason"]


def test_user_failure_is_not_caught_or_retried_as_a_compile_failure():
    calls = []

    def step(value):
        calls.append(1)
        raise ValueError("user failure")

    compiled = compile_training_callable(
        step, backend=lambda graph, examples: graph.forward, diagnostics={}
    )
    with pytest.raises(ValueError, match="user failure"):
        compiled(torch.tensor([0.1]))
    assert len(calls) == 1
