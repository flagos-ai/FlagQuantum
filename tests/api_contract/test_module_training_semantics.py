from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.training as fqt

pytestmark = pytest.mark.unit


def _build(parameters: torch.Tensor) -> fq.Circuit:
    return fq.Circuit(2).ry(0, parameters[0]).cx(0, 1).rx(1, parameters[1])


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_forward_and_execute_have_distinct_stable_return_types(mode: str) -> None:
    module = fq.Module(
        _build,
        2,
        init=torch.tensor([0.2, -0.3]),
        policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(mode=mode)),
    )

    value = module.forward()
    execution = module.execute()

    assert isinstance(value, torch.Tensor)
    assert isinstance(execution, fq.ExecutionResult)
    torch.testing.assert_close(value, execution.require_value())
    value.sum().backward()
    assert module.parameters_tensor.grad is not None


def test_module_is_not_an_implicit_run_program_or_batch_product() -> None:
    module = fq.Module(_build, 2)

    assert not hasattr(fq.Module, "run")
    with pytest.raises(TypeError):
        fq.run(module)

    batched_parameters = torch.zeros((2, 2))
    batched_inputs = torch.zeros((3, 1))
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        module.execute(batched_inputs, batched_parameters)


def test_parameter_override_does_not_replace_owned_parameters() -> None:
    module = fq.Module(_build, 2, init=torch.tensor([0.2, -0.3]))
    owned_before = module.parameters_tensor.detach().clone()
    override = torch.tensor([0.7, 0.4], requires_grad=True)

    execution = module.execute(parameters=override)
    value = module.forward(parameters=override)

    torch.testing.assert_close(value, execution.require_value())
    torch.testing.assert_close(module.parameters_tensor, owned_before)
    value.sum().backward()
    assert override.grad is not None


def test_result_diagnostics_and_required_value_fail_closed() -> None:
    empty = fq.ExecutionResult()

    with pytest.raises(RuntimeError, match="does not contain a module value"):
        empty.require_value()
    assert empty.diagnostics() == {
        "schema": "flagquantum.execution_diagnostics",
        "version": "1.0",
        "metrics": {},
        "provenance": {},
        "runtime": {},
        "compatibility": {},
    }


def test_train_is_a_minimal_caller_owned_optimizer_loop() -> None:
    module = fq.Module(_build, 2, init=torch.tensor([0.2, -0.3]))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    callbacks: list[tuple[int, float, fq.ExecutionResult]] = []

    result = fq.train(
        module,
        optimizer=optimizer,
        objective=lambda value: value.mean(),
        steps=2,
        callback=lambda step, loss, execution: callbacks.append(
            (step, loss, execution)
        ),
    )

    assert result.completed_steps == 2
    assert result.final_loss == result.losses[-1]
    assert result.summary()["schema"] == "flagquantum.training_result.summary"
    assert result.summary()["version"] == "1.0"
    assert [item[0] for item in callbacks] == [1, 2]
    assert all(item[2].value is not None for item in callbacks)
    assert all(item[2].value.grad_fn is None for item in callbacks)
    with pytest.raises(TypeError, match="unexpected keyword"):
        fq.train(
            module,
            optimizer=optimizer,
            objective=lambda value: value.mean(),
            steps=1,
            checkpoint="training.pt",  # type: ignore[call-arg]
        )


def test_training_result_invariants_fail_closed() -> None:
    execution = fq.ExecutionResult(value=torch.tensor(0.0))

    with pytest.raises(ValueError, match="completed_steps must be positive"):
        fq.TrainingResult((), 0, execution, "SGD")
    with pytest.raises(ValueError, match="loss count must equal completed_steps"):
        fq.TrainingResult((1.0,), 2, execution, "SGD")
    with pytest.raises(ValueError, match="optimizer must be non-empty"):
        fq.TrainingResult((1.0,), 1, execution, "")


def test_training_namespace_is_stable_without_root_clutter() -> None:
    assert fqt.PrecisionPolicy.__module__.endswith("training_state")
    assert fqt.SeedContract.__module__.endswith("training_state")
    assert fqt.TrainingCheckpointRestore.__module__.endswith("training_state")
    assert "TrainingCheckpointRestore" in fqt.__all__
    with pytest.raises(AttributeError):
        getattr(fq, "TrainingCheckpointRestore")
    for name in (
        "assert_finite_training",
        "load_training_checkpoint",
        "save_training_checkpoint",
    ):
        with pytest.raises(AttributeError):
            getattr(fq, name)
