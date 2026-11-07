"""Contract checks for explicit outputs on the local dynamic path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.observables import OutputRequest
from flagquantum.runtime import plan_execution
from flagquantum.runtime.dynamic import execution
from flagquantum.runtime.dynamic.result import DynamicExecutionResult
from flagquantum.runtime.execution_plan_contract import plan_program

pytestmark = pytest.mark.unit


@pytest.fixture
def feedback() -> DynamicCircuit:
    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    return circuit


def test_approved_output_kinds_survive_plan_round_trip(
    feedback: DynamicCircuit,
) -> None:
    contract = json.loads(
        (
            Path(__file__).parents[2] / "contracts/dynamic-final-outputs-v1.json"
        ).read_text()
    )
    options = fq.ExecutionOptions(shots=8, seed=7)
    plan = fq.plan(
        feedback,
        options=options,
        outputs=(fq.counts(name="histogram"), fq.samples(name="final")),
    )
    restored = type(plan).from_json(plan.to_json())
    assert plan_program(restored).instructions == feedback.to_ir().instructions
    result = fq.run(restored)
    assert tuple(item.kind for item in result.measurements) == (
        "counts",
        "sample",
        "mid_circuit_measurement",
    )
    assert result.measurement("histogram").value == result.counts
    assert contract["admission"]["output_kinds"] == ["samples", "counts"]
    assert result.measurement("final").kind == "sample"
    assert sum(result.counts[0].values()) == 8
    assert (
        result.measurement("mid_circuit_measurement").kind == "mid_circuit_measurement"
    )


@pytest.mark.parametrize(
    "outputs",
    [
        fq.samples((0,)),
        fq.counts((1, 0)),
        fq.samples(fq.X(0)),
        fq.expectation(fq.Z(0)),
        (fq.samples(), fq.samples(name="repeat")),
        fq.samples(name="mid_circuit_measurement"),
    ],
)
def test_rejects_outputs_outside_slice_before_execution(
    feedback: DynamicCircuit, outputs: OutputRequest | tuple[OutputRequest, ...]
) -> None:
    with pytest.raises((CapabilityError, ValueError), match="dynamic"):
        fq.plan(
            feedback,
            options=(
                None
                if isinstance(outputs, OutputRequest) and outputs.kind == "expectation"
                else fq.ExecutionOptions(shots=8)
            ),
            outputs=outputs,
        )


def test_explicit_terminal_measurement_is_refused(feedback: DynamicCircuit) -> None:
    feedback.measure(1, classical_bit=1)
    with pytest.raises(CapabilityError, match="explicit terminal measurement"):
        fq.plan(feedback, options=fq.ExecutionOptions(shots=8), outputs=fq.counts())


def test_inconsistent_replay_is_refused_before_sampling(
    feedback: DynamicCircuit, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = fq.plan(feedback, options=fq.ExecutionOptions(shots=8), outputs=fq.samples())
    ir = plan_program(plan)
    malformed = replace(ir, measurements=(replace(ir.measurements[0], shots=9),))
    monkeypatch.setattr(plan_execution, "plan_program", lambda _: malformed)

    def must_not_run() -> None:
        pytest.fail("inconsistent replay reached the dynamic sampler")

    monkeypatch.setattr(execution, "run_dynamic", must_not_run)
    with pytest.raises(ValidationError, match="shots disagree"):
        fq.run(plan)


def test_result_projection_uses_existing_samples(
    feedback: DynamicCircuit, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = torch.tensor([[1, 0], [0, 1], [1, 0]])
    bits = torch.tensor([[0], [1], [0]])

    def fake_run(
        circuit: DynamicCircuit, *, shots: int, seed: int | None, max_batched_bytes: int
    ) -> DynamicExecutionResult:
        assert shots == 3
        assert circuit.to_ir().instructions == feedback.to_ir().instructions
        return DynamicExecutionResult(samples, bits, torch.empty(0), shots, seed)

    monkeypatch.setattr(execution, "run_dynamic", fake_run)
    result = fq.run(feedback, outputs=(fq.samples(), fq.counts()), shots=3)
    assert result.samples is samples
    assert result.measurement("sample").value is samples
    assert result.measurement("mid_circuit_measurement").value is bits
    assert result.counts == [{"10": 2, "01": 1}]


def test_multibatch_explicit_outputs_are_refused() -> None:
    circuit = DynamicCircuit(2, bsz=2).h(0).measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    with pytest.raises(CapabilityError, match="single circuit batch"):
        fq.run(circuit, outputs=fq.counts(), shots=8)
