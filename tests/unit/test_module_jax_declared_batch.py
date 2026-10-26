"""Declared Circuit batches keep the existing Module broadcasting/VJP contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import ValidationError

pytestmark = [
    pytest.mark.unit,
    pytest.mark.jax,
    pytest.mark.skipif(
        importlib.util.find_spec("jax") is None, reason="jax is not installed"
    ),
]


def build(parameters: torch.Tensor, features: torch.Tensor) -> fq.Circuit:
    angle = parameters[0] + parameters[1] * features[:, 0]
    other = features[:, 1] + parameters[0] * parameters[0]
    return (
        fq.Circuit(2, bsz=features.shape[0])
        .ry(0, angle)
        .rz(0, parameters[1])
        .ry(1, other)
        .cx(0, 1)
    )


def model(mode: str, dtype: torch.dtype = torch.float64) -> fq.Module:
    return fq.Module(
        build,
        2,
        init=torch.tensor([0.23, -0.31], dtype=dtype),
        dtype=dtype,
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(backend="jax", mode=mode),
            observable="z_sum",
            observable_qubits=(0, 1),
        ),
    )


def analytic(parameters: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    angle = parameters[0] + parameters[1] * features[:, 0]
    other = features[:, 1] + parameters[0].square()
    return angle.cos() * (1 + other.cos())


@pytest.mark.parametrize("mode", ["statevector", "tensor_network", "mps"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 8, 64])
def test_declared_batch_handles_all_derivative_demands(
    mode: str,
    dtype: torch.dtype,
    batch: int,
) -> None:
    module = model(mode, dtype)
    parameters = module.parameters_tensor.detach().clone()
    features = torch.linspace(-0.37, 0.43, batch * 4, dtype=dtype).reshape(batch, 4)[
        :, ::2
    ]
    weights = torch.linspace(-0.7, 1.3, batch, dtype=dtype)
    tolerance = 3e-5 if dtype == torch.float32 else 1e-10
    for parameter_grad, input_grad in (
        (True, True),
        (False, True),
        (True, False),
        (False, False),
        (True, True),
    ):
        parameters.requires_grad_(parameter_grad)
        features.requires_grad_(input_grad)
        value = module(features, parameters=parameters)
        expected = analytic(parameters, features)
        assert value.shape == (batch,)
        torch.testing.assert_close(value, expected, atol=tolerance, rtol=tolerance)
        selected = tuple(
            tensor for tensor in (parameters, features) if tensor.requires_grad
        )
        if selected:
            actual = torch.autograd.grad((weights * value).sum(), selected)
            oracle = torch.autograd.grad((weights * expected).sum(), selected)
            for gradient, reference in zip(actual, oracle, strict=True):
                torch.testing.assert_close(
                    gradient, reference, atol=tolerance, rtol=tolerance
                )
        else:
            assert not value.requires_grad
        with torch.no_grad():
            parameters.add_(0.01)
            features.add_(0.005)
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            value = module(features, parameters=parameters)
            assert not value.requires_grad
            torch.testing.assert_close(
                value, analytic(parameters, features), atol=tolerance, rtol=tolerance
            )
    assert module._jax_kernel.summary()["declared_batch_size"] == batch


@pytest.mark.parametrize("mode", ["statevector", "tensor_network"])
def test_batch_adam_matches_analytic_training_and_checkpoint_replay(
    mode: str,
    tmp_path: Path,
) -> None:
    module = model(mode)
    reference = module.parameters_tensor.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam(module.parameters(), lr=0.03)
    reference_optimizer = torch.optim.Adam([reference], lr=0.03)
    features = torch.linspace(-0.3, 0.5, 16, dtype=torch.float64).reshape(8, 2)
    target = torch.linspace(-0.2, 0.4, 8, dtype=torch.float64)
    saved = None
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        value = module(features)
        expected = analytic(reference, features)
        torch.testing.assert_close(value, expected, atol=1e-10, rtol=1e-10)
        (value - target).square().mean().backward()
        (expected - target).square().mean().backward()
        torch.testing.assert_close(
            module.parameters_tensor.grad, reference.grad, atol=1e-10, rtol=1e-10
        )
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(
            module.parameters_tensor, reference, atol=1e-10, rtol=1e-10
        )
        if step == 2:
            saved = module.save_checkpoint(
                tmp_path / "batch.pt", optimizer=optimizer, seed=11, step=3
            )
    final = module.parameters_tensor.detach().clone()
    assert saved is not None
    restored = model(mode)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.03)
    restored(features)
    restored.load_checkpoint(saved, optimizer=restored_optimizer)
    for _ in range(2):
        restored_optimizer.zero_grad(set_to_none=True)
        (restored(features) - target).square().mean().backward()
        restored_optimizer.step()
    torch.testing.assert_close(
        restored.parameters_tensor, final, atol=1e-12, rtol=1e-12
    )
    for key, value in optimizer.state_dict()["state"][0].items():
        other = restored_optimizer.state_dict()["state"][0][key]
        torch.testing.assert_close(value, other, atol=1e-12, rtol=1e-12)


def test_batch_vjp_and_ir_keep_consumed_values_after_inplace_updates(
    tmp_path: Path,
) -> None:
    module = model("tensor_network")
    features = (
        torch.linspace(-0.3, 0.5, 16, dtype=torch.float64)
        .reshape(8, 2)
        .requires_grad_()
    )
    module(features)
    value = module(features)
    frozen_params = module.parameters_tensor.detach().clone().requires_grad_()
    frozen_features = features.detach().clone().requires_grad_()
    weights = torch.linspace(-0.4, 1.1, 8, dtype=torch.float64)
    expected = torch.autograd.grad(
        (weights * analytic(frozen_params, frozen_features)).sum(),
        (frozen_params, frozen_features),
    )
    reference = model("tensor_network")
    expected_hash = reference.execute(
        frozen_features, parameters=frozen_params
    ).provenance["ir_hash"]
    with torch.no_grad():
        module.parameters_tensor.add_(0.7)
        features.add_(0.4)
    actual = torch.autograd.grad(
        (weights * value).sum(), (module.parameters_tensor, features)
    )
    for gradient, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, oracle, atol=1e-10, rtol=1e-10)
    path = module.save_checkpoint(tmp_path / "consumed.pt", optimizer=None, seed=11)
    assert torch.load(path, weights_only=True)["ir_hash"] == expected_hash


def test_batch_bridge_does_not_retain_a_full_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax

    def forbidden(*args, **kwargs):
        raise AssertionError("batch bridging must use a VJP, not a full Jacobian")

    monkeypatch.setattr(jax, "jacrev", forbidden)
    monkeypatch.setattr(jax, "jacfwd", forbidden)
    module = model("statevector")
    features = (
        torch.linspace(-0.3, 0.5, 128, dtype=torch.float64)
        .reshape(64, 2)
        .requires_grad_()
    )
    module(features).sum().backward()
    saved_sizes = []

    def pack(value: torch.Tensor) -> torch.Tensor:
        saved_sizes.append(value.numel())
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        module(features).square().sum().backward()
    assert saved_sizes
    assert max(saved_sizes) <= features.numel()
    assert sum(saved_sizes) <= 4 * (features.numel() + 2 + 64)


def test_bad_declared_batch_shape_is_refused() -> None:
    def bad(parameters: torch.Tensor, inputs: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1, bsz=8).ry(0, parameters[0] + inputs)

    module = fq.Module(
        bad,
        1,
        policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(backend="jax")),
    )
    with pytest.raises(ValidationError, match="batch of 8"):
        module(torch.zeros(7))
