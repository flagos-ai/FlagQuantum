"""Named binding is replaceable while ordinary PyTorch training stays intact."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.errors import ValidationError
from flagquantum.runtime.builder_compilation import CompiledInstruction

pytestmark = pytest.mark.unit


def _core_bind(
    instructions: Iterable[Instruction], values: Mapping[str, torch.Tensor]
) -> tuple[Instruction, ...]:
    """Replace the Runtime binder with the existing public Circuit binder."""
    template = fq.Circuit(1)
    template._instructions.extend(instructions)
    return tuple(template.bind_parameters(values)._instructions)


@pytest.fixture(params=("runtime", "core"))
def named_binder(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.param == "core":
        monkeypatch.setattr(
            "flagquantum.runtime.training.module._bind_named_instructions", _core_bind
        )
        monkeypatch.setattr(
            "flagquantum.runtime._runtime_inputs._bind_named_instructions", _core_bind
        )


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
@pytest.mark.parametrize("batch", (1, 8))
def test_named_template_adam_and_checkpoint_recovery(
    named_binder: None, mode: str, batch: int, tmp_path: Path
) -> None:
    theta, bias = fq.Parameter("theta"), fq.Parameter("bias")
    source = fq.Circuit(1, bsz=batch, dtype=torch.complex128).ry(
        0, theta * theta + 0.25 * theta + bias
    )
    policy = fq.RuntimePolicy(execution_options=fq.ExecutionOptions(mode=mode))

    def make_model() -> fq.Module:
        return fq.Module(
            source,
            parameters={"theta": (batch,), "bias": ()},
            init={
                "theta": torch.linspace(0.3, 0.6, batch, dtype=torch.float64),
                "bias": 0.1,
            },
            dtype=torch.float64,
            policy=policy,
        )

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    reference = {
        name: value.detach().clone().requires_grad_()
        for name, value in model.parameter_groups.items()
    }
    reference_optimizer = torch.optim.Adam(reference.values(), lr=0.03)
    target = torch.linspace(-0.1, 0.2, batch, dtype=torch.float64)
    resumed: tuple[fq.Module, torch.optim.Adam] | None = None
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        prediction = model()
        expected = (
            reference["theta"].square() + 0.25 * reference["theta"] + reference["bias"]
        ).cos()
        torch.testing.assert_close(prediction, expected)
        loss = torch.nn.functional.mse_loss(prediction, target)
        torch.autograd.backward(loss)
        torch.autograd.backward(torch.nn.functional.mse_loss(expected, target))
        for name, value in model.parameter_groups.items():
            torch.testing.assert_close(value.grad, reference[name].grad)
        optimizer.step()
        reference_optimizer.step()
        for name, value in model.parameter_groups.items():
            torch.testing.assert_close(value, reference[name])
        if resumed is not None:
            replay, replay_optimizer = resumed
            replay_optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(torch.nn.functional.mse_loss(replay(), target))
            replay_optimizer.step()
            for name, value in model.parameter_groups.items():
                torch.testing.assert_close(value, replay.parameter_groups[name])
                for key, state in optimizer.state[value].items():
                    torch.testing.assert_close(
                        state,
                        replay_optimizer.state[replay.parameter_groups[name]][key],
                    )
        if step == 1:
            checkpoint = model.save_checkpoint(
                tmp_path / "named.pt", optimizer=optimizer, seed=11, step=2
            )
            replay = make_model()
            replay_optimizer = torch.optim.Adam(replay.parameters(), lr=0.03)
            replay()
            replay.load_checkpoint(checkpoint, optimizer=replay_optimizer)
            resumed = replay, replay_optimizer
    assert source.parameter_names == ("bias", "theta")
    assert set(model.state_dict()) == {
        "named_parameter_groups.bias",
        "named_parameter_groups.theta",
        "_extra_state",
    }


def test_run_and_module_keep_live_graphs_and_higher_derivatives(
    named_binder: None,
) -> None:
    theta = fq.Parameter("theta")
    source = fq.Circuit(1, dtype=torch.complex128).ry(0, theta * theta + 0.1)
    model = fq.Module(source, dtype=torch.float64)
    angles = [
        torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (0.3, 0.7)
    ]
    first = fq.run(source, inputs={"theta": angles[0]})
    assert isinstance(first.plan, fq.ExecutionPlan)
    plan = first.plan
    sealed = plan.to_json()
    # Keep all forwards alive before backward. Rebinding must not replace an
    # earlier call's graph, and parameter overrides must stay caller-owned.
    modules = [model(parameters={"theta": angle}) for angle in angles]
    runs = [first, fq.run(plan, inputs={"theta": angles[1]})]
    owned = model.parameter_groups["theta"].detach().clone()
    for angle, value, result in zip(angles, modules, runs, strict=True):
        expected = (angle.square() + 0.1).cos()
        state = result.statevector()[0]
        z = state[0].abs().square() - state[1].abs().square()
        for output in (value.sum(), z):
            torch.testing.assert_close(output, expected)
            derivative = torch.autograd.grad(output, angle, create_graph=True)[0]
            torch.testing.assert_close(
                derivative, -2 * angle * (angle.square() + 0.1).sin()
            )
            curvature = torch.autograd.grad(derivative, angle)[0]
            torch.testing.assert_close(
                curvature,
                -2 * (angle.square() + 0.1).sin()
                - 4 * angle.square() * (angle.square() + 0.1).cos(),
            )
    torch.testing.assert_close(model.parameter_groups["theta"], owned)
    assert model.parameter_groups["theta"].grad is None
    assert plan.to_json() == sealed
    assert source.parameter_names == ("theta",)


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
@pytest.mark.parametrize("batch", (1, 8))
def test_hybrid_model_uses_normal_pytorch_training(mode: str, batch: int) -> None:
    def build(groups: Mapping[str, torch.Tensor], features: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1, bsz=features.shape[0]).ry(
            0, groups["angle"].sin() + groups["scale"] * features[:, 0]
        )

    encoder = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[0.2, -0.3]], dtype=torch.float64))
        encoder.bias.fill_(0.1)
    quantum = fq.Module(
        build,
        parameters={"angle": (), "scale": ()},
        init={"angle": 0.3, "scale": 0.7},
        dtype=torch.float64,
        policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(mode=mode)),
    )
    model = torch.nn.Sequential(encoder, quantum)
    reference = [
        parameter.detach().clone().requires_grad_() for parameter in model.parameters()
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    reference_optimizer = torch.optim.Adam(reference, lr=0.03)
    weights, bias, angle, scale = reference
    for step in range(4):
        x = (
            torch.linspace(-0.5, 0.7, 2 * batch, dtype=torch.float64).reshape(batch, 2)
            + 0.01 * step
        ).requires_grad_()
        rx = x.detach().clone().requires_grad_()
        target = torch.linspace(-0.1, 0.2, batch, dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        expected = (angle.sin() + scale * (rx @ weights.T + bias)[:, 0]).cos()
        torch.testing.assert_close(prediction, expected)
        torch.autograd.backward(torch.nn.functional.mse_loss(prediction, target))
        torch.autograd.backward(torch.nn.functional.mse_loss(expected, target))
        torch.testing.assert_close(x.grad, rx.grad)
        for parameter, expected_parameter in zip(
            model.parameters(), reference, strict=True
        ):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad)
        optimizer.step()
        reference_optimizer.step()
        for parameter, expected_parameter in zip(
            model.parameters(), reference, strict=True
        ):
            torch.testing.assert_close(parameter, expected_parameter)
    assert quantum._builder_compile_count == 1
    model(x[:1])
    assert quantum._builder_compile_count == (1 if batch == 1 else 2)
    assert set(quantum.state_dict()) == {
        "named_parameter_groups.angle",
        "named_parameter_groups.scale",
        "_extra_state",
    }
    with pytest.raises(RuntimeError, match="not bound"):
        quantum._builder_bindings.values()


def test_overflow_is_rejected_after_a_successful_call(named_binder: None) -> None:
    source = fq.Circuit(1, dtype=torch.complex128).ry(0, 2 * fq.Parameter("theta"))
    model = fq.Module(source, dtype=torch.float64)
    model(parameters={"theta": torch.tensor(0.2, dtype=torch.float64)})
    first = fq.run(source, inputs={"theta": 0.2})
    with pytest.raises(ValidationError, match="finite"):
        model(parameters={"theta": torch.tensor(1e308, dtype=torch.float64)})
    with pytest.raises(ValidationError, match="finite"):
        fq.run(first.plan, inputs={"theta": 1e308})
    with pytest.raises(TypeError, match="does not accept inputs"):
        model(torch.zeros(1, 1, dtype=torch.float64))


def test_circuit_subclass_retains_its_binding_behavior() -> None:
    class InvertedReadout(Circuit):
        def bind_parameters(self, values: Mapping[str, torch.Tensor]) -> fq.Circuit:
            return super().bind_parameters(values).x(0)

    source = InvertedReadout(1, dtype=torch.complex128).ry(0, fq.Parameter("theta"))
    model = fq.Module(source, init={"theta": 0.4}, dtype=torch.float64)
    torch.testing.assert_close(
        model(), -model.parameter_groups["theta"].cos().reshape(1)
    )


def test_reimported_compiled_instruction_is_validated_when_bound(
    named_binder: None,
) -> None:
    instruction = CompiledInstruction(
        "ry", (0,), {"theta": 2 * fq.Parameter("theta")}, None, {}, (0,), (None,)
    )
    source = fq.Circuit.from_ir(fq.CircuitIR(1, (instruction,), dtype="complex128"))
    model = fq.Module(source, dtype=torch.float64)
    with pytest.raises(ValidationError, match="finite"):
        model(parameters={"theta": torch.tensor(1e308, dtype=torch.float64)})
    with pytest.raises(ValidationError, match="finite"):
        fq.run(source, inputs={"theta": 1e308})
    torch.testing.assert_close(
        model(parameters={"theta": torch.tensor(0.2, dtype=torch.float64)}),
        torch.tensor([0.4], dtype=torch.float64).cos(),
    )
