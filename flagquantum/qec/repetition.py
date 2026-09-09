"""Three-qubit repetition-code reference workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..noise import NoiseModel
from ..runtime.dynamic._feedback import (
    DynamicFeedbackAction,
    DynamicFeedbackObservation,
    DynamicFeedbackPlan,
    DynamicFeedbackPoint,
)
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .decoders import (
    Decoder,
    RepetitionLookupDecoder,
    RepetitionStreamingLookupDecoder,
    RepetitionTemporalDecoder,
    StreamingDecoder,
)
from .types import (
    Correction,
    DecodeResult,
    DetectionEvent,
    ErrorSchedule,
    PauliFrame,
    RepetitionMemoryResult,
    RepetitionMemoryShot,
    SyndromeRound,
)

_FEEDBACK_MODES = {
    "compiled_lookup",
    "offline_pauli_frame",
    "runtime_decoder",
    "runtime_pauli_frame",
    "runtime_temporal_decoder",
    "runtime_temporal_pauli_frame",
}


def _adjust_for_frame(
    bits: tuple[int, int], frame_x_wires: tuple[int, ...]
) -> tuple[int, int]:
    frame = set(frame_x_wires)
    return (
        bits[0] ^ int((0 in frame) != (1 in frame)),
        bits[1] ^ int((1 in frame) != (2 in frame)),
    )


@dataclass(frozen=True)
class _RepetitionFeedbackController:
    decoder: StreamingDecoder
    action_mode: str

    def decide(
        self, history: Sequence[DynamicFeedbackObservation]
    ) -> DynamicFeedbackAction:
        previous = (0, 0)
        syndrome_history = []
        for observation in history:
            if len(observation.observed_bits) != 2:
                raise ValueError("repetition feedback requires two observed bits")
            raw = (observation.observed_bits[0], observation.observed_bits[1])
            bits = _adjust_for_frame(raw, observation.frame_x_wires_before)
            round_index = observation.decision_index
            events = tuple(
                DetectionEvent(round_index, check_index)
                for check_index, (before, after) in enumerate(zip(previous, bits))
                if before != after
            )
            syndrome_history.append(SyndromeRound(round_index, bits, events))
            previous = bits
        correction = self.decoder.decode_round(tuple(syndrome_history))
        if not isinstance(correction, Correction):
            raise TypeError("streaming decoder must return a Correction")
        if correction.round_index != len(syndrome_history) - 1:
            raise ValueError("streaming correction must belong to the current round")
        if correction.wire is None:
            return DynamicFeedbackAction()
        return DynamicFeedbackAction(mode=self.action_mode, wire=correction.wire)


def _feedback_plan(
    rounds: int,
    decoder: StreamingDecoder,
    *,
    action_mode: str,
) -> DynamicFeedbackPlan:
    return DynamicFeedbackPlan(
        points=tuple(
            DynamicFeedbackPoint(
                name=f"repetition_round_{round_index}",
                trigger_classical_bit=2 * round_index + 1,
                classical_bits=(2 * round_index, 2 * round_index + 1),
            )
            for round_index in range(rounds)
        ),
        controller=_RepetitionFeedbackController(decoder, action_mode),
        allowed_wires=(0, 1, 2),
        allowed_action_modes=(action_mode,),
    )


def _memory_source(schedule: ErrorSchedule, *, compiled_feedback: bool) -> str:
    lines = [
        "def repetition_memory(rounds):",
        "    left = False",
        "    right = False",
        "    for round_index in range(rounds):",
    ]
    for event in schedule.events:
        lines.extend(
            (
                f"        if round_index == {event.round_index}:",
                f"            qp.X(wires={event.wire})",
            )
        )
    lines.extend(
        (
            "        qp.CNOT(wires=[0, 3])",
            "        qp.CNOT(wires=[1, 3])",
            "        left = qp.measure(wires=3)",
            "        qp.reset(wires=3)",
            "        qp.CNOT(wires=[1, 4])",
            "        qp.CNOT(wires=[2, 4])",
            "        right = qp.measure(wires=4)",
            "        qp.reset(wires=4)",
        )
    )
    if compiled_feedback:
        lines.extend(
            (
                "        if left and not right:",
                "            qp.X(wires=0)",
                "        elif left and right:",
                "            qp.X(wires=1)",
                "        elif not left and right:",
                "            qp.X(wires=2)",
            )
        )
    lines.append("    return right")
    return "\n".join(lines) + "\n"


def _syndrome_rounds(
    values: Sequence[int], *, rounds: int
) -> tuple[SyndromeRound, ...]:
    if len(values) != rounds * 2:
        raise ValueError("classical register does not match repetition-code rounds")
    previous = (0, 0)
    records = []
    for round_index in range(rounds):
        bits = (int(values[2 * round_index]), int(values[2 * round_index + 1]))
        events = tuple(
            DetectionEvent(round_index, check_index)
            for check_index, (before, after) in enumerate(zip(previous, bits))
            if before != after
        )
        records.append(SyndromeRound(round_index, bits, events))
        previous = bits
    return tuple(records)


def _lookup_correction(record: SyndromeRound) -> Correction:
    wire = {
        (0, 0): None,
        (1, 0): 0,
        (1, 1): 1,
        (0, 1): 2,
    }[record.bits]
    return Correction(round_index=record.round_index, wire=wire)


def run_repetition_memory_experiment(
    *,
    error_schedule: ErrorSchedule | None = None,
    rounds: int = 3,
    shots: int = 128,
    seed: int | None = 0,
    strategy: str = "auto",
    feedback_mode: str = "compiled_lookup",
    decoder: Decoder | None = None,
    feedback_decoder: StreamingDecoder | None = None,
    noise_model: NoiseModel | None = None,
) -> RepetitionMemoryResult:
    """Run the bounded three-qubit bit-flip memory reference.

    ``compiled_lookup`` keeps feedback in the lowered circuit. The two
    ``runtime_*`` modes call a replaceable decoder after every syndrome round
    and apply either a physical X or a tracked Pauli-frame X. The offline mode
    applies only the terminal analysis frame to recorded readout.
    """

    schedule = error_schedule or ErrorSchedule()
    if not isinstance(schedule, ErrorSchedule):
        raise TypeError("error_schedule must be an ErrorSchedule or None")
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if feedback_mode not in _FEEDBACK_MODES:
        raise ValueError("unsupported repetition feedback mode")
    if any(event.round_index >= rounds for event in schedule.events):
        raise ValueError("error event is outside the configured rounds")
    selected_decoder = RepetitionLookupDecoder() if decoder is None else decoder
    if not isinstance(selected_decoder, Decoder):
        raise TypeError("decoder must implement the Decoder protocol")
    if feedback_decoder is None:
        selected_feedback_decoder = (
            RepetitionTemporalDecoder()
            if feedback_mode.startswith("runtime_temporal_")
            else RepetitionStreamingLookupDecoder()
        )
    else:
        selected_feedback_decoder = feedback_decoder
    if feedback_mode.startswith("runtime_") and not isinstance(
        selected_feedback_decoder, StreamingDecoder
    ):
        raise TypeError("feedback_decoder must implement the StreamingDecoder protocol")

    source = _memory_source(
        schedule,
        compiled_feedback=feedback_mode == "compiled_lookup",
    )
    program = capture_source(source, (INDEX,))
    lowered = lower_dynamic_program(
        program,
        (rounds,),
        max_dynamic_measurements=rounds * 2,
    )
    runtime_plan = None
    if feedback_mode in {"runtime_decoder", "runtime_temporal_decoder"}:
        runtime_plan = _feedback_plan(
            rounds, selected_feedback_decoder, action_mode="physical_x"
        )
    elif feedback_mode in {
        "runtime_pauli_frame",
        "runtime_temporal_pauli_frame",
    }:
        runtime_plan = _feedback_plan(
            rounds, selected_feedback_decoder, action_mode="frame_x"
        )
    execution = execute_hybrid_dynamic_session(
        lowered.circuit,
        shots=shots,
        seed=seed,
        strategy=strategy,
        noise_model=noise_model,
        _feedback_plan=runtime_plan,
    )
    classical_rows = execution.classical_bits.tolist()
    sample_rows = execution.samples.tolist()
    if len(classical_rows) != shots or len(sample_rows) != shots:
        raise RuntimeError("dynamic execution returned an unexpected shot count")

    shot_records = []
    for shot_index, (classical, sample) in enumerate(zip(classical_rows, sample_rows)):
        syndrome_rounds = _syndrome_rounds(classical, rounds=rounds)
        if feedback_mode == "compiled_lookup":
            executed_feedback = tuple(
                _lookup_correction(record) for record in syndrome_rounds
            )
        elif feedback_mode.startswith("runtime_"):
            trace = execution.feedback_traces[shot_index]
            executed_feedback = tuple(
                Correction(
                    round_index=decision.observation.decision_index,
                    wire=decision.action.wire,
                )
                for decision in trace.decisions
            )
        else:
            executed_feedback = tuple(
                Correction(round_index=record.round_index, wire=None)
                for record in syndrome_rounds
            )
        decode_result = selected_decoder.decode(syndrome_rounds)
        if not isinstance(decode_result, DecodeResult):
            raise TypeError("decoder must return a DecodeResult")
        final_data = tuple(int(value) for value in sample[:3])
        if feedback_mode in {
            "runtime_pauli_frame",
            "runtime_temporal_pauli_frame",
        }:
            runtime_frame = PauliFrame.from_corrections(executed_feedback)
            raw_final_data = runtime_frame.apply(final_data)
            decoded_data = final_data
        else:
            raw_final_data = final_data
            decoded_data = (
                decode_result.pauli_frame.apply(raw_final_data)
                if feedback_mode == "offline_pauli_frame"
                else raw_final_data
            )
        logical_bit = int(sum(decoded_data) >= 2)
        shot_records.append(
            RepetitionMemoryShot(
                shot_index=shot_index,
                syndrome_rounds=syndrome_rounds,
                executed_feedback=executed_feedback,
                decode_result=decode_result,
                raw_final_data_bits=raw_final_data,
                decoded_data_bits=decoded_data,
                logical_bit=logical_bit,
                logical_failure=bool(logical_bit),
            )
        )
    return RepetitionMemoryResult(
        rounds=rounds,
        error_schedule=schedule,
        feedback_mode=feedback_mode,
        shot_records=tuple(shot_records),
        execution_semantics=execution.execution_semantics,
        noise_model_identity=execution.statistics["noise_model_identity"],
        bit_flip_events=int(execution.statistics["bit_flip_event_count"]),
        readout_errors=int(execution.statistics["readout_error_count"]),
    )


__all__ = ("run_repetition_memory_experiment",)
