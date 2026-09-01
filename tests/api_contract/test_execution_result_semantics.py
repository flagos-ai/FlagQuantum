from __future__ import annotations

import json

import pytest
import torch

import flagquantum as fq
import flagquantum.noise as fqn

pytestmark = pytest.mark.unit


def test_explicit_measurements_are_embedded_in_the_executable_plan() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    request = fq.MeasurementNode(
        "probabilities", (0, 1), metadata={"name": "bell_probabilities"}
    )

    plan = fq.plan(circuit, measurements=(request,))
    restored = fq.ExecutionPlan.from_json(plan.to_json())
    result = fq.run(restored)

    assert restored.program_fingerprint == plan.program_fingerprint
    assert result.plan is restored
    torch.testing.assert_close(
        result.measurement("bell_probabilities").value,
        torch.tensor([[0.5, 0.0, 0.0, 0.5]]),
    )


def test_explicit_measurements_never_overwrite_program_measurements() -> None:
    program = fq.CircuitIR(
        n_wires=1,
        instructions=(),
        measurements=(fq.MeasurementNode("probabilities", (0,)),),
    )

    with pytest.raises(ValueError, match="already contains measurement requests"):
        fq.plan(
            program,
            measurements=(fq.MeasurementNode("expectation_z", (0,)),),
        )


def test_noise_model_is_verified_serialized_and_executed_from_the_plan() -> None:
    circuit = fq.Circuit(1).x(0)
    noise_model = fqn.NoiseModel().add("x", fqn.bit_flip_channel(1.0))

    plan = fq.plan(circuit, noise_model=noise_model)
    payload = plan.to_dict()
    restored = fq.ExecutionPlan.from_dict(payload)
    result = fq.run(restored)

    assert payload["extensions"][0]["identity"] == noise_model.identity
    assert restored.identity == plan.identity
    assert restored.summary() == plan.summary()
    assert result.plan is restored
    torch.testing.assert_close(
        result.state[:, 0, 0], torch.ones(1, dtype=torch.complex64)
    )

    tampered = json.loads(plan.to_json())
    tampered["extensions"][0]["identity"] = "0" * 64
    with pytest.raises(ValueError, match="extension_incompatible"):
        fq.ExecutionPlan.from_dict(tampered)


def test_result_accessors_are_explicit_and_backend_attributes_do_not_leak() -> None:
    circuit = fq.Circuit(1).x(0)
    result = fq.run(
        circuit,
        measurements=(
            fq.MeasurementNode("expectation_z", (0,), metadata={"name": "energy"}),
        ),
    )

    torch.testing.assert_close(result.expectation(), torch.tensor([[-1.0]]))
    assert result.measurement("energy") is result.measurement(0)
    assert result.statevector() is result.state
    assert result.native() is result.state
    assert result.summary()["schema"] == "flagquantum.execution_result.summary"
    assert result.summary()["version"] == "1.0"
    with pytest.raises(RuntimeError, match="does not contain samples"):
        result.require_samples()
    with pytest.raises(AttributeError):
        _ = result.shape


def test_result_measurement_selector_fails_closed_on_missing_or_ambiguous_data() -> (
    None
):
    result = fq.ExecutionResult(
        measurements=(
            fq.MeasurementResult("expectation_z", (0,), torch.ones(1)),
            fq.MeasurementResult("expectation_z", (1,), torch.zeros(1)),
        )
    )

    with pytest.raises(RuntimeError, match="ambiguous"):
        result.measurement("expectation_z")
    with pytest.raises(RuntimeError, match="multiple expectations"):
        result.expectation()
    with pytest.raises(RuntimeError, match="no measurement"):
        result.measurement("missing")
