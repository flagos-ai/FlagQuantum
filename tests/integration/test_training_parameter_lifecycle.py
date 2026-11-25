"""Current public APIs support explicit parameter lifecycle decisions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.local.train_parameters import main

pytestmark = pytest.mark.integration


def _model(mode: str = "statevector") -> fq.Module:
    def build(parameters: Mapping[str, torch.Tensor]) -> fq.Circuit:
        return fq.Circuit(1).ry(0, parameters["encoder"]).rx(0, parameters["readout"])

    return fq.Module(
        build,
        parameters={"encoder": (), "readout": ()},
        init={"encoder": 0.4, "readout": 0.7},
        dtype=torch.float64,
        policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(mode=mode)),
    )


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
@pytest.mark.parametrize("optimizer_kind", ("AdamW", "SGD"))
def test_freeze_after_backward_clears_stale_gradient_and_preserves_history(
    mode: str, optimizer_kind: str
) -> None:
    model = _model(mode)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.1)
        if optimizer_kind == "AdamW"
        else torch.optim.SGD(
            model.parameters(), lr=0.03, momentum=0.8, weight_decay=0.1
        )
    )
    encoder = model.parameter_groups["encoder"]
    readout = model.parameter_groups["readout"]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(model().square().mean())
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(model().square().mean())
    assert encoder.grad is not None
    assert bool(encoder.grad.abs() > 0)
    encoder.requires_grad_(False)
    encoder.grad = None
    frozen = encoder.detach().clone()
    before_readout = readout.detach().clone()
    history = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in optimizer.state[encoder].items()
    }
    optimizer.step()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(model().square().mean())
        assert encoder.grad is None
        optimizer.step()
    torch.testing.assert_close(encoder, frozen, atol=0, rtol=0)
    torch.testing.assert_close(optimizer.state[encoder], history, atol=0, rtol=0)
    assert not torch.equal(readout, before_readout)

    encoder.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(model().square().mean())
    optimizer.step()
    assert not torch.equal(encoder, frozen)
    assert len(optimizer.param_groups) == 1


def test_unfreezing_does_not_implicitly_add_optimizer_membership() -> None:
    model = _model()
    encoder = model.parameter_groups["encoder"]
    encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW([model.parameter_groups["readout"]], lr=0.03)
    before = encoder.detach().clone()
    encoder.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    torch.autograd.backward(model().square().mean())
    optimizer.step()
    assert encoder.grad is not None
    torch.testing.assert_close(encoder, before, atol=0, rtol=0)

    optimizer.add_param_group({"params": [encoder]})
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(model().square().mean())
    optimizer.step()
    assert not torch.equal(encoder, before)


def test_initializer_copies_values_and_training_flags_are_explicit() -> None:
    initial = torch.tensor(0.4, dtype=torch.float64, requires_grad=False)
    model = fq.Module(
        lambda groups: fq.Circuit(1).ry(0, groups["angle"]),
        parameters={"angle": initial},
        dtype=torch.float64,
    )
    parameter = model.parameter_groups["angle"]
    torch.testing.assert_close(parameter, initial)
    assert parameter.data_ptr() != initial.data_ptr()
    assert parameter.requires_grad
    parameter.requires_grad_(False)
    with torch.no_grad():
        initial.add_(0.1)
    torch.testing.assert_close(parameter, torch.tensor(0.4, dtype=torch.float64))


def test_zero_stationary_point_and_explicit_nonzero_start() -> None:
    def build(weights: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1).ry(0, weights[0])

    zero = fq.Module(build, 1, dtype=torch.float64)
    nonzero = fq.Module(build, 1, init=torch.tensor([0.4]), dtype=torch.float64)
    for model in (zero, nonzero):
        torch.autograd.backward(model().square().mean())
    assert zero.parameters_tensor is not None
    assert nonzero.parameters_tensor is not None
    torch.testing.assert_close(
        zero.parameters_tensor.grad, torch.zeros(1, dtype=torch.float64)
    )
    assert nonzero.parameters_tensor.grad is not None
    assert bool(nonzero.parameters_tensor.grad.abs().sum() > 0)


def test_complete_current_api_example() -> None:
    main()


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
@pytest.mark.parametrize("kind", ("SGD", "AdamW"))
def test_checkpoint_resumes_freeze_and_unfreeze_trajectory(
    tmp_path: Path, mode: str, kind: str
) -> None:
    source, target = _model(mode), _model(mode)
    optimizer_type = getattr(torch.optim, kind)
    left, right = optimizer_type(source.parameters(), lr=0.03), optimizer_type(
        target.parameters(), lr=0.09
    )
    for _ in range(2):
        left.zero_grad(set_to_none=True)
        source().square().mean().backward()
        left.step()
    source.parameter_groups["encoder"].requires_grad_(False)
    source.parameter_groups["encoder"].grad = None
    path = source.save_checkpoint(
        tmp_path / "resume.pt", optimizer=left, seed=11, step=2
    )
    target.load_checkpoint(path, optimizer=right)
    assert not target.parameter_groups["encoder"].requires_grad
    for index in range(4):
        if index == 2:
            source.parameter_groups["encoder"].requires_grad_(True)
            target.parameter_groups["encoder"].requires_grad_(True)
        for model, optimizer in ((source, left), (target, right)):
            optimizer.zero_grad(set_to_none=True)
            model().square().mean().backward()
            optimizer.step()
        for first, second in zip(source.parameters(), target.parameters(), strict=True):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(
            left.state_dict(), right.state_dict(), atol=0, rtol=0
        )
