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
    RepetitionNoiseProfile,
    RepetitionStreamingLookupDecoder,
    RepetitionTemporalDecoder,
    SyndromeRound,
    run_repetition_memory_experiment,
    run_repetition_memory_noise_sweep,
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


@pytest.mark.parametrize("feedback_mode", ("runtime_decoder", "runtime_pauli_frame"))
@pytest.mark.parametrize("error_round", (0, 1, 2))
@pytest.mark.parametrize("wire", (0, 1, 2))
def test_runtime_streaming_feedback_corrects_single_error(
    feedback_mode: str, error_round: int, wire: int
) -> None:
    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(error_round, wire),)),
        rounds=3,
        shots=2,
        seed=97,
        strategy="trajectory",
        feedback_mode=feedback_mode,
    )

    assert result.logical_error_rate == 0.0
    for shot in result.shot_records:
        assert shot.executed_feedback[error_round].wire == wire
        assert shot.raw_final_data_bits == tuple(
            int(feedback_mode == "runtime_pauli_frame" and index == wire)
            for index in range(3)
        )
        assert shot.decoded_data_bits == (0, 0, 0)


def test_replacing_streaming_decoder_changes_runtime_execution() -> None:
    class WrongWireDecoder:
        def decode_round(self, syndrome_history) -> Correction:
            return Correction(len(syndrome_history) - 1, 1)

    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(0, 0),)),
        rounds=1,
        shots=1,
        seed=101,
        strategy="trajectory",
        feedback_mode="runtime_decoder",
        feedback_decoder=WrongWireDecoder(),
    )

    shot = result.shot_records[0]
    assert shot.executed_feedback == (Correction(0, 1),)
    assert shot.raw_final_data_bits == (1, 1, 0)
    assert shot.logical_failure


def test_streaming_lookup_decoder_consumes_available_history() -> None:
    decoder = RepetitionStreamingLookupDecoder()
    history = (SyndromeRound(0, (0, 0)), SyndromeRound(1, (0, 1)))

    assert decoder.decode_round(history) == Correction(1, 2)


def test_temporal_decoder_confirms_persistent_data_syndrome() -> None:
    decoder = RepetitionTemporalDecoder()
    onset = SyndromeRound(0, (1, 0), (DetectionEvent(0, 0),))
    confirmed = SyndromeRound(1, (1, 0))

    assert decoder.decode_round((onset,)) == Correction(0, None)
    assert decoder.decode_round((onset, confirmed)) == Correction(1, 0)


def test_temporal_decoder_rejects_isolated_readout_syndrome() -> None:
    decoder = RepetitionTemporalDecoder()
    false_onset = SyndromeRound(0, (0, 1), (DetectionEvent(0, 1),))
    cleared = SyndromeRound(1, (0, 0), (DetectionEvent(1, 1),))

    assert decoder.decode_round((false_onset, cleared)) == Correction(1, None)


@pytest.mark.parametrize(
    "feedback_mode",
    ("runtime_temporal_decoder", "runtime_temporal_pauli_frame"),
)
@pytest.mark.parametrize("error_round", (0, 1))
@pytest.mark.parametrize("wire", (0, 1, 2))
def test_temporal_runtime_feedback_corrects_confirmable_data_errors(
    feedback_mode: str, error_round: int, wire: int
) -> None:
    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(error_round, wire),)),
        rounds=3,
        shots=2,
        seed=107,
        strategy="trajectory",
        feedback_mode=feedback_mode,
    )

    for shot in result.shot_records:
        assert shot.executed_feedback[error_round].wire is None
        assert shot.executed_feedback[error_round + 1].wire == wire
        assert shot.decoded_data_bits == (0, 0, 0)
        assert not shot.logical_failure


@pytest.mark.parametrize(
    "feedback_mode",
    ("runtime_temporal_decoder", "runtime_temporal_pauli_frame"),
)
def test_temporal_runtime_leaves_terminal_round_error_unconfirmed(
    feedback_mode: str,
) -> None:
    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(2, 0),)),
        rounds=3,
        shots=1,
        seed=109,
        strategy="trajectory",
        feedback_mode=feedback_mode,
    )

    shot = result.shot_records[0]
    assert all(not correction.applied for correction in shot.executed_feedback)
    assert shot.raw_final_data_bits == (1, 0, 0)
    assert shot.syndrome_rounds[-1].bits == (1, 0)


@pytest.mark.parametrize("wire", (0, 1, 2))
def test_compiled_runtime_and_frame_modes_agree_on_logical_outcome(wire: int) -> None:
    schedule = ErrorSchedule((ErrorEvent(1, wire),))
    results = tuple(
        run_repetition_memory_experiment(
            error_schedule=schedule,
            rounds=3,
            shots=4,
            seed=103,
            strategy="trajectory",
            feedback_mode=mode,
        )
        for mode in ("compiled_lookup", "runtime_decoder", "runtime_pauli_frame")
    )

    assert tuple(result.logical_failures for result in results) == (0, 0, 0)
    assert all(
        all(shot.decoded_data_bits == (0, 0, 0) for shot in result.shot_records)
        for result in results
    )


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
    with pytest.raises(ValueError, match="requires syndrome history"):
        RepetitionStreamingLookupDecoder().decode_round(())
    with pytest.raises(ValueError, match="requires syndrome history"):
        RepetitionTemporalDecoder().decode_round(())
    with pytest.raises(ValueError, match="inconsistent with syndrome history"):
        RepetitionTemporalDecoder().decode_round((SyndromeRound(0, (1, 0)),))
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


def test_repetition_noise_profile_uses_existing_noise_model_contract() -> None:
    profile = RepetitionNoiseProfile(
        data_bit_flip_probability=0.2,
        syndrome_readout_error_probability=0.1,
        final_readout_error_probability=0.05,
    )
    model = profile.to_noise_model()

    assert len(model.rules) == 3
    assert tuple(rule.wires for rule in model.rules) == ((0,), (1,), (2,))
    assert tuple(rule.wires for rule in model.readout_rules) == ((3, 4), (0, 1, 2))
    with pytest.raises(ValueError, match="between zero and one"):
        RepetitionNoiseProfile(data_bit_flip_probability=1.1)


def test_stochastic_repetition_noise_is_seeded_and_auditable() -> None:
    model = RepetitionNoiseProfile(data_bit_flip_probability=0.2).to_noise_model()

    first = run_repetition_memory_experiment(
        rounds=3,
        shots=64,
        seed=71,
        strategy="batched",
        noise_model=model,
    )
    second = run_repetition_memory_experiment(
        rounds=3,
        shots=64,
        seed=71,
        strategy="batched",
        noise_model=model,
    )

    assert first.noise_model_identity == model.identity
    assert first.bit_flip_events > 0
    assert first.bit_flip_events == second.bit_flip_events
    assert first.logical_failures == second.logical_failures
    assert tuple(shot.syndrome_rounds for shot in first.shot_records) == tuple(
        shot.syndrome_rounds for shot in second.shot_records
    )


def test_syndrome_readout_noise_changes_observed_feedback() -> None:
    model = RepetitionNoiseProfile(
        syndrome_readout_error_probability=1.0
    ).to_noise_model()
    result = run_repetition_memory_experiment(
        rounds=3,
        shots=2,
        seed=73,
        strategy="batched",
        noise_model=model,
    )

    assert all(shot.syndrome_rounds[0].bits == (1, 1) for shot in result.shot_records)
    assert result.readout_errors == 16


def test_temporal_decoder_reduces_spurious_feedback_in_seeded_readout_noise() -> None:
    model = RepetitionNoiseProfile(
        syndrome_readout_error_probability=0.1
    ).to_noise_model()
    immediate = run_repetition_memory_experiment(
        rounds=4,
        shots=128,
        seed=113,
        strategy="trajectory",
        feedback_mode="runtime_decoder",
        noise_model=model,
    )
    temporal = run_repetition_memory_experiment(
        rounds=4,
        shots=128,
        seed=113,
        strategy="trajectory",
        feedback_mode="runtime_temporal_decoder",
        noise_model=model,
    )

    immediate_actions = sum(
        correction.applied
        for shot in immediate.shot_records
        for correction in shot.executed_feedback
    )
    temporal_actions = sum(
        correction.applied
        for shot in temporal.shot_records
        for correction in shot.executed_feedback
    )
    assert immediate.readout_errors > 0
    assert temporal.readout_errors > 0
    assert temporal_actions < immediate_actions


def test_noise_sweep_records_finite_shot_points_without_suppression_claim() -> None:
    points = run_repetition_memory_noise_sweep(
        (0.0, 0.2),
        rounds=3,
        shots=64,
        seed=79,
        strategy="batched",
    )

    assert points[0].logical_error_rate == 0.0
    assert points[0].bit_flip_events == 0
    assert points[1].bit_flip_events > 0
    assert points[1].logical_error_rate == points[1].logical_failures / 64
    with pytest.raises(ValueError, match="at least one probability"):
        run_repetition_memory_noise_sweep(())
