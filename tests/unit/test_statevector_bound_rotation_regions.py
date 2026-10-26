"""Compiled angle packing preserves ordinary Module training and broadcasting."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import ValidationError
from flagquantum.simulation.statevector import batching
from flagquantum.simulation.statevector.program import _StatevectorFusedGateStep

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width,depth", [(2, 1), (2, 3), (4, 1), (4, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 4])
def test_packed_regions_preserve_shared_parameter_adam_updates(
    width: int,
    depth: int,
    dtype: torch.dtype,
    batch: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(parameters: torch.Tensor, features: torch.Tensor) -> fq.Circuit:
        circuit = fq.Circuit(
            width,
            bsz=batch,
            dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
        )
        for layer in range(depth):
            for wire in range(width):
                index = 2 * (layer * width + wire)
                circuit.ry(wire, parameters[index].sin() + features[wire])
                circuit.rz(wire, parameters[index + 1] * parameters[0])
            for wire in range(width - 1):
                circuit.cx(wire, wire + 1)
        return circuit

    initial = torch.linspace(-0.2, 0.3, 2 * width * depth, dtype=dtype)
    models = [
        fq.Module(build, initial.numel(), init=initial.clone(), dtype=dtype)
        for _ in range(2)
    ]
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.03) for model in models]
    pack = batching._bound_scalar_rotation_angles
    packed_calls = []

    def observed(
        steps: Sequence[_StatevectorFusedGateStep],
        state: torch.Tensor,
        bindings: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor | None:
        result = pack(steps, state, bindings)
        if result is not None:
            packed_calls.append(result.shape)
        return result

    monkeypatch.setattr(batching, "_bound_scalar_rotation_angles", observed)
    for step in range(3):
        outcomes = []
        for index, (model, optimizer) in enumerate(
            zip(models, optimizers, strict=True)
        ):
            features = torch.linspace(
                0.1 + step * 0.03, 0.4, width, dtype=dtype
            ).requires_grad_()
            with monkeypatch.context() if index else nullcontext() as context:
                if context is not None:
                    context.setattr(
                        batching, "_bound_scalar_rotation_angles", lambda *args: None
                    )
                optimizer.zero_grad(set_to_none=True)
                value = model(features)
                weights = torch.arange(1, value.numel() + 1, dtype=dtype).reshape(
                    value.shape
                )
                (value.square() * weights).sum().backward()
                outcomes.append(
                    (
                        value.detach(),
                        model.parameters_tensor.grad.clone(),
                        features.grad.clone(),
                    )
                )
                optimizer.step()
        for actual, expected in zip(*outcomes, strict=True):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            models[0].parameters_tensor, models[1].parameters_tensor
        )
        assert models[0]._last_ir.content_hash == models[1]._last_ir.content_hash
    assert packed_calls


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_packed_regions_work_without_autograd(
    context: Callable[[], AbstractContextManager[object]],
) -> None:
    def build(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1).ry(0, parameters[0]).rz(0, parameters[1])

    model = fq.Module(build, 2, init=torch.tensor([0.3, 0.2]), dtype=torch.float64)
    with context():
        actual = model()
    assert not actual.requires_grad
    torch.testing.assert_close(actual.reshape(()), model.parameters_tensor[0].cos())


@pytest.mark.parametrize("kind", ["constant", "vector", "cast", "invalid_vector"])
def test_other_parameter_forms_preserve_values_gradients_and_shape_refusal(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(parameters: torch.Tensor, features: torch.Tensor) -> fq.Circuit:
        angle = features if kind in {"vector", "invalid_vector"} else features[0]
        if kind == "cast":
            angle = angle.float()
        else:
            angle = angle + parameters[0]
        return (
            fq.Circuit(1, bsz=3, dtype=torch.complex128)
            .ry(0, angle)
            .rz(0, 0.2 if kind == "constant" else parameters[1])
        )

    features = torch.tensor(
        [0.1, 0.2] if kind == "invalid_vector" else [0.1, 0.2, 0.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    parameters = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    model = fq.Module(build, 2, dtype=torch.float64)
    if kind == "invalid_vector":
        with pytest.raises(ValidationError, match="length 2"):
            model(features, parameters=parameters)
        with monkeypatch.context() as context:
            context.setattr(
                batching, "_bound_scalar_rotation_angles", lambda *args: None
            )
            with pytest.raises(ValidationError, match="length 2"):
                model(features, parameters=parameters)
        return
    actual = model(features, parameters=parameters)
    actual_grads = torch.autograd.grad(actual.square().sum(), (parameters, features))
    with monkeypatch.context() as context:
        context.setattr(batching, "_bound_scalar_rotation_angles", lambda *args: None)
        expected = model(features, parameters=parameters)
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (parameters, features)
        )
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
