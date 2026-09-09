"""Three-qubit repetition-code reference workflow."""

from __future__ import annotations

from typing import Sequence

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .decoders import Decoder, RepetitionLookupDecoder
from .types import (
    Correction,
    DecodeResult,
    DetectionEvent,
    ErrorSchedule,
    RepetitionMemoryResult,
    RepetitionMemoryShot,
    SyndromeRound,
)

_FEEDBACK_MODES = {"compiled_lookup", "offline_pauli_frame"}


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
) -> RepetitionMemoryResult:
    """Run the bounded three-qubit bit-flip memory reference.

    ``compiled_lookup`` applies immediate reference feedback in the dynamic
    program. ``offline_pauli_frame`` leaves the quantum data uncorrected and
    applies the decoder's terminal Pauli frame to the recorded readout. Both
    are development evidence over a noiseless local simulator.
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
    execution = execute_hybrid_dynamic_session(
        lowered.circuit,
        shots=shots,
        seed=seed,
        strategy=strategy,
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
        else:
            executed_feedback = tuple(
                Correction(round_index=record.round_index, wire=None)
                for record in syndrome_rounds
            )
        decode_result = selected_decoder.decode(syndrome_rounds)
        if not isinstance(decode_result, DecodeResult):
            raise TypeError("decoder must return a DecodeResult")
        raw_final_data = tuple(int(value) for value in sample[:3])
        decoded_data = (
            raw_final_data
            if feedback_mode == "compiled_lookup"
            else decode_result.pauli_frame.apply(raw_final_data)
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
    )


__all__ = ("run_repetition_memory_experiment",)
