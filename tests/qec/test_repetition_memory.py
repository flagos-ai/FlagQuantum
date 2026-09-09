from __future__ import annotations

import pytest

from flagquantum.qec import (
    Correction,
    Decoder,
    DecodeResult,
    DetectionEvent,
    ErrorEvent,
    ErrorSchedule,
    PauliFrame,
    RepetitionLookupDecoder,
    RepetitionMemoryShot,
    SyndromeRound,
    run_repetition_memory_experiment,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("syndrome", "wire"),
    (
        ((0, 0), None),
        ((1, 0), 0),
        ((1, 1), 1),
        ((0, 1), 2),
    ),
)
def test_repetition_lookup_decoder_uses_complete_history(syndrome, wire) -> None:
    decoder = RepetitionLookupDecoder()
    history = (SyndromeRound(0, (0, 0)), SyndromeRound(1, syndrome))

    assert isinstance(decoder, Decoder)
    decoded = decoder.decode(history)
    assert decoded.consumed_rounds == 2
    assert decoded.corrections == (Correction(round_index=1, wire=wire),)
    assert decoded.pauli_frame.x_wires == (() if wire is None else (wire,))


@pytest.mark.parametrize("error_round", (0, 1, 2))
@pytest.mark.parametrize(("wire", "syndrome"), ((0, (1, 0)), (1, (1, 1)), (2, (0, 1))))
@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_compiled_feedback_corrects_single_error_in_any_round(
    error_round: int,
    wire: int,
    syndrome: tuple[int, int],
    strategy: str,
) -> None:
    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(error_round, wire),)),
        rounds=3,
        shots=2,
        seed=53,
        strategy=strategy,
        feedback_mode="compiled_lookup",
    )

    assert result.logical_error_rate == 0.0
    for shot in result.shot_records:
        expected = [(0, 0), (0, 0), (0, 0)]
        expected[error_round] = syndrome
        assert [record.bits for record in shot.syndrome_rounds] == expected
        assert shot.executed_feedback[error_round].wire == wire
        assert shot.raw_final_data_bits == (0, 0, 0)
        assert shot.decoded_data_bits == (0, 0, 0)


@pytest.mark.parametrize("wire", (0, 1, 2))
def test_offline_pauli_frame_restores_single_error_without_physical_feedback(
    wire: int,
) -> None:
    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(1, wire),)),
        rounds=3,
        shots=2,
        seed=59,
        feedback_mode="offline_pauli_frame",
    )

    for shot in result.shot_records:
        expected_raw = tuple(int(index == wire) for index in range(3))
        assert shot.raw_final_data_bits == expected_raw
        assert shot.decode_result.pauli_frame.x_wires == (wire,)
        assert shot.decoded_data_bits == (0, 0, 0)
        assert all(not item.applied for item in shot.executed_feedback)
        assert not shot.logical_failure


@pytest.mark.parametrize("feedback_mode", ("compiled_lookup", "offline_pauli_frame"))
def test_two_same_round_errors_are_recorded_as_a_logical_failure(
    feedback_mode: str,
) -> None:
    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(1, 0), ErrorEvent(1, 1))),
        rounds=3,
        shots=1,
        seed=61,
        feedback_mode=feedback_mode,
    )

    shot = result.shot_records[0]
    assert shot.decoded_data_bits == (1, 1, 1)
    assert shot.logical_failure
    assert result.logical_error_rate == 1.0


def test_detection_events_record_error_onset_and_compiled_clearance() -> None:
    shot = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(1, 1),)),
        rounds=3,
        shots=1,
        seed=67,
    ).shot_records[0]

    assert tuple(
        tuple(event.check_index for event in record.detection_events)
        for record in shot.syndrome_rounds
    ) == ((), (0, 1), (0, 1))


def test_error_schedule_is_canonical_and_decoder_is_replaceable() -> None:
    schedule = ErrorSchedule((ErrorEvent(1, 2), ErrorEvent(0, 0)))
    assert schedule.events == (ErrorEvent(0, 0), ErrorEvent(1, 2))
    assert schedule.for_round(1) == (ErrorEvent(1, 2),)

    class WireZeroDecoder:
        def decode(self, syndrome_history) -> DecodeResult:
            correction = Correction(len(syndrome_history) - 1, 0)
            return DecodeResult(
                consumed_rounds=len(syndrome_history),
                corrections=(correction,),
                pauli_frame=PauliFrame((0,)),
            )

    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(0, 0),)),
        rounds=2,
        shots=1,
        feedback_mode="offline_pauli_frame",
        decoder=WireZeroDecoder(),
    )
    assert result.shot_records[0].decode_result.pauli_frame == PauliFrame((0,))


def test_repetition_memory_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="wire must be 0, 1, or 2"):
        ErrorEvent(0, 3)
    with pytest.raises(ValueError, match="cannot repeat"):
        ErrorSchedule((ErrorEvent(0, 1), ErrorEvent(0, 1)))
    with pytest.raises(TypeError, match="must be ErrorEvent"):
        ErrorSchedule((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside the configured rounds"):
        run_repetition_memory_experiment(
            error_schedule=ErrorSchedule((ErrorEvent(3, 1),)), rounds=3
        )
    with pytest.raises(ValueError, match="unsupported repetition feedback mode"):
        run_repetition_memory_experiment(feedback_mode="realtime")
    with pytest.raises(ValueError, match="requires syndrome history"):
        RepetitionLookupDecoder().decode(())
    with pytest.raises(ValueError, match="check index exceeds syndrome width"):
        SyndromeRound(0, (0, 0), (DetectionEvent(0, 2),))
    with pytest.raises(ValueError, match="feedback rounds must align"):
        RepetitionMemoryShot(
            shot_index=0,
            syndrome_rounds=(SyndromeRound(0, (0, 0)),),
            executed_feedback=(Correction(1, None),),
            decode_result=DecodeResult(1, (Correction(0, None),), PauliFrame()),
            raw_final_data_bits=(0, 0, 0),
            decoded_data_bits=(0, 0, 0),
            logical_bit=0,
            logical_failure=False,
        )

    class InvalidDecoder:
        def decode(self, syndrome_history):
            return None

    with pytest.raises(TypeError, match="must return a DecodeResult"):
        run_repetition_memory_experiment(rounds=1, shots=1, decoder=InvalidDecoder())
