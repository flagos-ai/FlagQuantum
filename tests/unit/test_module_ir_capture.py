"""Classical binding lowering is checked against independent Torch arithmetic."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler.module_builder import capture_module_builder
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit

RECIPES = {
    "neg": lambda p, x: -x[:, 0],
    "sin": lambda p, x: x[:, 0].sin(),
    "cos": lambda p, x: x[:, 0].cos(),
    "tan": lambda p, x: x[:, 0].tan(),
    "tanh": lambda p, x: x[:, 0].tanh(),
    "exp": lambda p, x: x[:, 0].exp(),
    "log": lambda p, x: (x[:, 0] + 2).log(),
    "sqrt": lambda p, x: (x[:, 0] + 2).sqrt(),
    "sigmoid": lambda p, x: x[:, 0].sigmoid(),
    "add_alpha": lambda p, x: torch.add(x[:, 0], p[0], alpha=2),
    "sub_alpha": lambda p, x: torch.sub(x[:, 0], p[0], alpha=2),
    "mul": lambda p, x: x[:, 0] * p[0],
    "div": lambda p, x: x[:, 0] / (p[0] + 2),
    "power_scalar": lambda p, x: x[:, 0] ** 2,
    "power_tensor": lambda p, x: (x[:, 0] + 2).pow(p[0] + 2),
    "scalar_power": lambda p, x: torch.pow(2.0, x[:, 0]),
    "where_gt": lambda p, x: torch.where(x[:, 0] > 0, x[:, 1], p[0]),
    "where_ge": lambda p, x: torch.where(x[:, 0] >= 0, x[:, 1], p[0]),
    "where_lt": lambda p, x: torch.where(x[:, 0] < 0, x[:, 1], p[0]),
    "where_le": lambda p, x: torch.where(x[:, 0] <= 0, x[:, 1], p[0]),
    "where_eq": lambda p, x: torch.where(x[:, 0] == 0, x[:, 1], p[0]),
    "where_ne": lambda p, x: torch.where(x[:, 0] != 0, x[:, 1], p[0]),
    "relu": lambda p, x: x[:, 0].relu(),
    "detach": lambda p, x: x[:, 0].detach() + p[0],
    "view": lambda p, x: x.view(3, 2)[:, 0],
    "reshape_copy": lambda p, x: x.t().reshape(6)[:3],
    "permute": lambda p, x: x.permute(1, 0)[0],
    "transpose": lambda p, x: x.transpose(0, 1)[0],
    "squeeze": lambda p, x: x[:, 0].unsqueeze(0).squeeze(),
    "squeeze_nonunit": lambda p, x: x.squeeze(0)[:, 0],
    "expand": lambda p, x: p[0].expand(3) + x[:1, 0].expand(3),
    "stack": lambda p, x: torch.stack((x[:, 0], x[:, 1]), 0)[0] + p[0],
    "cat": lambda p, x: torch.cat((x, x), 0)[3:, 0] + p[0],
    "unbind": lambda p, x: x.unbind(1)[0] + x.unbind(1)[1] * p[0],
    "clone": lambda p, x: x[:, 0].clone(),
    "alias": lambda p, x: torch.ops.aten.alias.default(x[:, 0]),
    "zeros_like": lambda p, x: torch.zeros_like(x[:, 0]),
    "ones_like": lambda p, x: torch.ones_like(x[:, 0]),
    "explicit_cast": lambda p, x: x[:, 0].float().double() + p[0],
}


@pytest.mark.parametrize("recipe", RECIPES, ids=RECIPES)
def test_bound_angle_values_and_nonuniform_vjps_match_torch(recipe):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    function = RECIPES[recipe]
    p = torch.tensor([0.23, -0.31], dtype=torch.float64, requires_grad=True)
    x = torch.tensor(
        [[-0.3, 0.2], [0.0, 0.1], [0.5, -0.2]], dtype=torch.float64, requires_grad=True
    )
    calls = []

    def build(parameters, features):
        calls.append(1)
        return fq.Circuit(1, bsz=3).ry(0, function(parameters, features))

    capture = capture_module_builder(build, (p, x), names=(), has_inputs=True)
    lowered, reason = capture.jax_builder()
    assert lowered is not None, reason
    assert reason == "canonical_ir_fx_bindings"
    assert all(
        isinstance(item.params["theta"], Parameter)
        for item in capture.template.instructions
    )

    def angles(parameters, features):
        return lowered(parameters, features).to_ir().instructions[0].params["theta"]

    weights = torch.tensor([-0.7, 1.3, 0.2], dtype=torch.float64)
    expected = function(p, x)
    # Keep constant and detached recipes differentiable to expose their zero VJP.
    objective = (expected * weights).sum() + (p.sum() + x.sum()) * 0
    expected_grad = torch.autograd.grad(objective, (p, x))
    params, features = jnp.asarray(p.detach().numpy()), jnp.asarray(x.detach().numpy())
    actual = jax.jit(angles)(params, features)
    gradients = jax.jit(
        jax.grad(
            lambda a, b: jnp.sum(angles(a, b) * jnp.asarray(weights.numpy())),
            argnums=(0, 1),
        )
    )(params, features)
    import numpy as np

    np.testing.assert_allclose(
        actual, expected.detach().numpy(), atol=1e-10, rtol=1e-10
    )
    for a, b in zip(gradients, expected_grad, strict=True):
        np.testing.assert_allclose(
            a,
            b.detach().numpy(),
            atol=1e-7 if recipe == "explicit_cast" else 1e-10,
            rtol=1e-7 if recipe == "explicit_cast" else 1e-10,
        )
    assert calls == [1]


def test_unadmitted_namespace_and_mutable_binding_have_reasons():
    p = torch.tensor([0.2], dtype=torch.float64)
    x = torch.ones(3, 1, dtype=torch.float64)

    def foreign(parameters, features):
        value = torch.ops.prims.add.default(parameters[0].expand(3), features[:, 0])
        return fq.Circuit(1, bsz=3).ry(0, value)

    captured = capture_module_builder(foreign, (p, x), names=(), has_inputs=True)
    lowered, reason = captured.jax_builder()
    assert lowered is None and "prims.add" in reason

    def mutable(parameters, features):
        value = features[:, 0].clone()
        value.add_(parameters[0])
        return fq.Circuit(1, bsz=3).ry(0, value)

    captured = capture_module_builder(mutable, (p, x), names=(), has_inputs=True)
    lowered, reason = captured.jax_builder()
    assert lowered is None and "add_" in reason


def test_torch_binding_graph_preserves_a_user_gradient_cut():
    p = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    capture = capture_module_builder(
        lambda parameters, features: fq.Circuit(1).ry(
            0, parameters[0].detach() * features[0]
        ),
        (p, x),
        names=(),
        has_inputs=True,
    )
    (angle,) = capture.binding_graph(p, x)
    parameter_grad, input_grad = torch.autograd.grad(angle, (p, x), allow_unused=True)
    assert parameter_grad is None
    torch.testing.assert_close(input_grad, p.detach(), atol=1e-12, rtol=1e-12)


def test_named_group_layout_is_preserved_without_an_implicit_jax_flattening_policy():
    weights = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    features = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    capture = capture_module_builder(
        lambda p, x: fq.Circuit(1).ry(0, p["weights"][0] + p["bias"][0] * x[0]),
        (weights, bias, features),
        names=("weights", "bias"),
        has_inputs=True,
    )
    assert capture.group_names == ("weights", "bias")
    (angle,) = capture.binding_graph(weights, bias, features)
    torch.testing.assert_close(angle, weights[0] + bias[0] * features[0])
    gradients = torch.autograd.grad(angle, (weights, bias, features))
    for actual, expected in zip(
        gradients,
        (torch.ones_like(weights), features.detach(), bias.detach()),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    assert capture.jax_builder() == (None, "named_parameter_groups")
