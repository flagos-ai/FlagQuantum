from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.noise as fqn
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic._feedback import (
    DynamicFeedbackAction,
    DynamicFeedbackPlan,
    DynamicFeedbackPoint,
)

pytestmark = pytest.mark.integration


class _CopyObservedBit:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode

    def decide(self, history):
        observation = history[-1]
        if observation.observed_bits == (1,):
            return DynamicFeedbackAction(self.mode, 1)
        return DynamicFeedbackAction()


def _plan(*, mode: str) -> DynamicFeedbackPlan:
    return DynamicFeedbackPlan(
        points=(DynamicFeedbackPoint("after_measurement", 0, (0,)),),
        controller=_CopyObservedBit(mode=mode),
        allowed_wires=(1,),
        allowed_action_modes=(mode,),
    )


def test_physical_feedback_changes_continued_quantum_state_and_is_traced() -> None:
    circuit = DynamicCircuit(2).x(0)
    circuit.measure(0, classical_bit=0)

    result = fq.experimental.dynamic.run_dynamic(
        circuit,
        shots=3,
        seed=83,
        strategy="trajectory",
        _feedback_plan=_plan(mode="physical_x"),
    )

    assert torch.equal(result.samples, torch.ones((3, 2), dtype=torch.int64))
    assert torch.allclose(
        torch.abs(result.final_states[:, 3]), torch.ones(3, dtype=torch.float32)
    )
    assert result.statistics["feedback_decision_count"] == 3
    decision = result.feedback_traces[0].decisions[0]
    assert decision.observation.true_bits == (1,)
    assert decision.observation.observed_bits == (1,)
    assert decision.observation.frame_x_wires_before == ()
    assert decision.action == DynamicFeedbackAction("physical_x", 1)
    assert decision.frame_x_wires_after == ()


def test_frame_feedback_changes_readout_interpretation_not_quantum_state() -> None:
    circuit = DynamicCircuit(2).x(0)
    circuit.measure(0, classical_bit=0)

    result = fq.experimental.dynamic.run_dynamic(
        circuit,
        shots=2,
        seed=89,
        strategy="auto",
        _feedback_plan=_plan(mode="frame_x"),
    )

    assert torch.equal(result.samples, torch.ones((2, 2), dtype=torch.int64))
    assert torch.allclose(
        torch.abs(result.final_states[:, 2]), torch.ones(2, dtype=torch.float32)
    )
    decision = result.feedback_traces[0].decisions[0]
    assert decision.observation.frame_x_wires_before == ()
    assert decision.frame_x_wires_after == (1,)


def test_feedback_trace_separates_true_and_observed_measurements() -> None:
    circuit = DynamicCircuit(2).x(0)
    circuit.measure(0, classical_bit=0)
    model = fqn.NoiseModel().add_readout(
        0,
        fqn.ReadoutError(((0.0, 1.0), (1.0, 0.0))),
    )

    result = fq.experimental.dynamic.run_dynamic(
        circuit,
        shots=1,
        seed=91,
        strategy="trajectory",
        noise_model=model,
        _feedback_plan=_plan(mode="physical_x"),
    )

    decision = result.feedback_traces[0].decisions[0]
    assert decision.observation.true_bits == (1,)
    assert decision.observation.observed_bits == (0,)
    assert decision.action == DynamicFeedbackAction()


def test_feedback_fails_closed_for_batched_and_out_of_plan_action() -> None:
    circuit = DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)

    with pytest.raises(ValueError, match="trajectory execution"):
        fq.experimental.dynamic.run_dynamic(
            circuit,
            shots=1,
            strategy="batched",
            _feedback_plan=_plan(mode="physical_x"),
        )

    class OutsidePlan:
        def decide(self, history):
            return DynamicFeedbackAction("physical_x", 0)

    invalid = DynamicFeedbackPlan(
        points=(DynamicFeedbackPoint("point", 0, (0,)),),
        controller=OutsidePlan(),
        allowed_wires=(1,),
    )
    with pytest.raises(ValueError, match="outside the plan"):
        fq.experimental.dynamic.run_dynamic(
            circuit, shots=1, strategy="trajectory", _feedback_plan=invalid
        )

    missing = DynamicFeedbackPlan(
        points=(DynamicFeedbackPoint("missing", 1, (1,)),),
        controller=_CopyObservedBit(mode="physical_x"),
        allowed_wires=(1,),
    )
    with pytest.raises(ValueError, match="does not name a measured bit"):
        fq.experimental.dynamic.run_dynamic(
            circuit, shots=1, strategy="trajectory", _feedback_plan=missing
        )
