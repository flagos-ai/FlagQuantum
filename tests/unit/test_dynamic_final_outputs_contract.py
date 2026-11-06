"""Contract checks for explicit outputs on the local dynamic path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.observables import OutputRequest

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
        feedback, options=options, outputs=(fq.samples(name="final"), fq.counts())
    )
    restored = type(plan).from_json(plan.to_json())
    result = fq.run(restored)
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
        fq.plan(feedback, options=fq.ExecutionOptions(shots=8), outputs=outputs)


def test_explicit_terminal_measurement_is_refused(feedback: DynamicCircuit) -> None:
    feedback.measure(1, classical_bit=1)
    with pytest.raises(CapabilityError, match="explicit terminal measurement"):
        fq.plan(feedback, options=fq.ExecutionOptions(shots=8), outputs=fq.counts())
