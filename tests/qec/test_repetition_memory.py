from __future__ import annotations

import pytest

from flagquantum.qec import (
    Correction,
    Decoder,
    DetectionEvent,
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
def test_repetition_lookup_decoder(syndrome, wire) -> None:
    decoder = RepetitionLookupDecoder()

    assert isinstance(decoder, Decoder)
    correction = decoder.decode(syndrome, round_index=2)
    assert correction.round_index == 2
    assert correction.wire == wire
    assert correction.applied is (wire is not None)


@pytest.mark.parametrize(
    ("error_wire", "first_syndrome"),
    ((0, (1, 0)), (1, (1, 1)), (2, (0, 1))),
)
@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_repetition_memory_corrects_each_single_data_error(
    error_wire: int, first_syndrome: tuple[int, int], strategy: str
) -> None:
    result = run_repetition_memory_experiment(
        error_wire=error_wire,
        rounds=3,
        shots=8,
        seed=53,
        strategy=strategy,
    )

    assert result.logical_error_rate == 0.0
    assert result.logical_failures == 0
    assert result.shot_count == 8
    for shot in result.shot_records:
        assert tuple(record.bits for record in shot.syndrome_rounds) == (
            first_syndrome,
            (0, 0),
            (0, 0),
        )
        assert shot.corrections[0].wire == error_wire
        assert tuple(item.wire for item in shot.corrections[1:]) == (None, None)
        assert shot.final_data_bits == (0, 0, 0)
        assert shot.logical_bit == 0
        assert not shot.logical_failure


def test_repetition_memory_reports_detection_events_and_no_error_baseline() -> None:
    corrected = run_repetition_memory_experiment(
        error_wire=1, rounds=3, shots=1, seed=59
    ).shot_records[0]
    assert tuple(
        tuple(event.check_index for event in record.detection_events)
        for record in corrected.syndrome_rounds
    ) == ((0, 1), (0, 1), ())

    baseline = run_repetition_memory_experiment(
        error_wire=None, rounds=2, shots=2, seed=61
    )
    assert baseline.injected_error_wire is None
    assert baseline.logical_error_rate == 0.0
    assert all(
        record.bits == (0, 0)
        for shot in baseline.shot_records
        for record in shot.syndrome_rounds
    )


def test_analysis_decoder_is_replaceable_without_changing_executed_feedback() -> None:
    class NoCorrectionAnalysis:
        def decode(self, syndrome, *, round_index: int) -> Correction:
            return Correction(round_index=round_index, wire=None)

    result = run_repetition_memory_experiment(
        error_wire=1,
        rounds=2,
        shots=1,
        analysis_decoder=NoCorrectionAnalysis(),
    )

    assert tuple(item.wire for item in result.shot_records[0].corrections) == (
        None,
        None,
    )
    assert result.shot_records[0].final_data_bits == (0, 0, 0)


def test_repetition_memory_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="0, 1, 2, or None"):
        run_repetition_memory_experiment(error_wire=3)
    with pytest.raises(ValueError, match="rounds must be a positive integer"):
        run_repetition_memory_experiment(error_wire=0, rounds=0)
    with pytest.raises(ValueError, match="two binary values"):
        RepetitionLookupDecoder().decode((1, 0, 1), round_index=0)
    with pytest.raises(ValueError, match="check index exceeds syndrome width"):
        SyndromeRound(0, (0, 0), (DetectionEvent(0, 2),))
    with pytest.raises(ValueError, match="correction rounds must align"):
        RepetitionMemoryShot(
            shot_index=0,
            syndrome_rounds=(SyndromeRound(0, (0, 0)),),
            corrections=(Correction(1, None),),
            final_data_bits=(0, 0, 0),
            logical_bit=0,
            logical_failure=False,
        )
