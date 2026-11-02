"""Default Module diagnostics preserve consumed values without eager copies."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit


def build(parameters: torch.Tensor, features: torch.Tensor) -> fq.Circuit:
    return (
        fq.Circuit(2, bsz=features.shape[0])
        .ry(0, parameters[0] + parameters[1] * features[:, 0])
        .ry(1, parameters[0].square() + features[:, 1])
        .cx(0, 1)
    )


def model(mode: str) -> fq.Module:
    return fq.Module(
        build,
        2,
        init=torch.tensor([0.23, -0.31]),
        dtype=torch.float64,
        policy=fq.RuntimePolicy(
            execution_options=fq.ExecutionOptions(mode=mode),
            observable="z_sum",
            observable_qubits=(0, 1),
        ),
    )


def analytic(parameters: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return (parameters[0] + parameters[1] * features[:, 0]).cos() * (
        1 + (parameters[0].square() + features[:, 1]).cos()
    )


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_local_values_gradients_and_consumed_hash_survive_updates(mode: str) -> None:
    module, detailed = model(mode), model(mode)
    features = (
        torch.linspace(-0.3, 0.5, 16, dtype=torch.float64)
        .reshape(8, 2)
        .requires_grad_()
    )
    weights = torch.linspace(-0.7, 1.3, 8, dtype=torch.float64)
    for _ in range(3):
        value = module(features)
        oracle = analytic(module.parameters_tensor, features)
        torch.testing.assert_close(value, oracle, atol=1e-10, rtol=1e-10)
        actual = torch.autograd.grad(
            (value * weights).sum(), (module.parameters_tensor, features)
        )
        expected = torch.autograd.grad(
            (oracle * weights).sum(), (module.parameters_tensor, features)
        )
        for gradient, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(gradient, reference, atol=1e-10, rtol=1e-10)
        with torch.no_grad():
            detailed.parameters_tensor.copy_(module.parameters_tensor)
        expected_hash = detailed.execute(features).provenance["ir_hash"]
        assert module._last_ir_snapshot is None
        with torch.no_grad():
            module.parameters_tensor.add_(0.01)
            features.add_(0.005)
        assert module._last_ir.content_hash == expected_hash


@pytest.mark.parametrize("inference", (False, True))
def test_snapshot_retains_the_consuming_gradient_context(inference: bool) -> None:
    module, detailed = model("statevector"), model("statevector")
    features = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    with torch.inference_mode() if inference else torch.no_grad():
        module(features)
        expected = detailed.execute(features).provenance["ir_hash"]
    assert module._last_ir.content_hash == expected


def test_checkpoint_keeps_preupdate_values_and_recompiles_after_dtype_change(
    tmp_path: Path,
) -> None:
    module, detailed = model("statevector"), model("statevector")
    optimizer = torch.optim.Adam(module.parameters(), lr=0.03)
    features = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    expected = detailed.execute(features).provenance["ir_hash"]
    module(features).sum().backward()
    optimizer.step()
    features.add_(0.4)
    path = module.save_checkpoint(
        tmp_path / "module.pt", optimizer=optimizer, seed=11, step=1
    )
    assert torch.load(path, weights_only=True)["ir_hash"] == expected
    module.float()
    module(features.float())
    assert module._last_ir.dtype == "complex64"


def test_large_unused_inputs_use_bounded_eager_slot_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sparse(parameters: torch.Tensor, features: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1).ry(0, parameters[0] + features[0])

    module = fq.Module(sparse, 1)
    features = torch.zeros(4096)
    module(features)
    original = torch.Tensor.clone

    def clone(
        value: torch.Tensor,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> torch.Tensor:
        assert value.numel() < features.numel()
        return original(value, memory_format=memory_format)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    module(features).sum().backward()
    assert module._last_ir_snapshot is not None


def test_nonfinite_binding_refusal_keeps_last_consumed_snapshot() -> None:
    module = model("statevector")
    features = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    module(features)
    expected = module._last_ir.content_hash
    with torch.no_grad():
        module.parameters_tensor.fill_(float("nan"))
    with pytest.raises(fq.errors.ValidationError):
        module(features)
    assert module._last_ir.content_hash == expected


def test_default_higher_order_derivative_remains_an_ordinary_torch_graph() -> None:
    module = fq.Module(
        lambda p: fq.Circuit(1).ry(0, p[0].square()),
        1,
        init=torch.tensor([0.2]),
        dtype=torch.float64,
    )
    p = module.parameters_tensor
    value = module().sum()
    gradient = torch.autograd.grad(value, p, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient.sum(), p)[0]
    expected = -2 * torch.sin(
        p.detach().square()
    ) - 4 * p.detach().square() * torch.cos(p.detach().square())
    torch.testing.assert_close(hessian, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_adam_trajectory_and_restored_state_match_analytic_reference(
    mode: str, tmp_path: Path
) -> None:
    module = model(mode)
    parameters = module.parameters_tensor.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam(module.parameters(), lr=0.03)
    reference_optimizer = torch.optim.Adam([parameters], lr=0.03)
    restored = None
    for step in range(6):
        features = (
            torch.linspace(-0.3, 0.5, 16, dtype=torch.float64).reshape(8, 2)
            + step * 0.01
        )
        target = torch.linspace(-0.2, 0.4, 8, dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        value = module(features)
        expected = analytic(parameters, features)
        torch.testing.assert_close(value, expected, atol=1e-10, rtol=1e-10)
        (value - target).square().mean().backward()
        (expected - target).square().mean().backward()
        torch.testing.assert_close(
            module.parameters_tensor.grad, parameters.grad, atol=1e-10, rtol=1e-10
        )
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(
            module.parameters_tensor, parameters, atol=1e-10, rtol=1e-10
        )
        if restored is not None:
            replay, replay_optimizer = restored
            replay_optimizer.zero_grad(set_to_none=True)
            (replay(features) - target).square().mean().backward()
            replay_optimizer.step()
            torch.testing.assert_close(
                replay.parameters_tensor,
                module.parameters_tensor,
                atol=1e-10,
                rtol=1e-10,
            )
            for key, value in optimizer.state_dict()["state"][0].items():
                torch.testing.assert_close(
                    value,
                    replay_optimizer.state_dict()["state"][0][key],
                    atol=1e-10,
                    rtol=1e-10,
                )
        if step == 2:
            path = module.save_checkpoint(
                tmp_path / "adam.pt", optimizer=optimizer, seed=11, step=3
            )
            replay = model(mode)
            replay_optimizer = torch.optim.Adam(replay.parameters(), lr=0.03)
            replay(features)
            replay.load_checkpoint(path, optimizer=replay_optimizer)
            restored = replay, replay_optimizer
