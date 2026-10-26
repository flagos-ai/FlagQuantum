"""Fixed-shot statistics against score-test inversion and controlled batches."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, Mock

import pytest
import torch

import examples.qec.stim_pymatching_baseline as baseline
from flagquantum.errors import CapabilityError
from flagquantum.qec import DemSample, MemorySampler, PyMatchingDecoder

pytestmark = pytest.mark.unit
_CASES = [
    (0, 100),
    (100, 100),
    (1, 100),
    (7, 50),
    (50, 100),
    (0, 1),
    (1, 1),
    (31, 35),
    (500_000, 1_000_000),
]


def _invert_score_test(failures: int, shots: int) -> tuple[float, float]:
    """Numerically invert the score inequality, without the closed-form formula."""
    z = 1.959963984540054
    observed = failures / shots

    def accepted(probability: float) -> bool:
        return (failures - shots * probability) ** 2 <= z**2 * shots * probability * (
            1 - probability
        )

    lower, upper = 0.0, observed
    for _ in range(80):
        middle = (lower + upper) / 2
        if accepted(middle):
            upper = middle
        else:
            lower = middle
    left_endpoint = upper
    lower, upper = observed, 1.0
    for _ in range(80):
        middle = (lower + upper) / 2
        if accepted(middle):
            lower = middle
        else:
            upper = middle
    return left_endpoint, lower


@pytest.mark.parametrize(("failures", "shots"), _CASES)
def test_wilson_interval_matches_independent_score_inversion(
    failures: int, shots: int
) -> None:
    interval = baseline._wilson_interval(failures, shots)
    assert interval == pytest.approx(_invert_score_test(failures, shots), abs=1e-12)
    assert 0 <= interval[0] <= failures / shots <= interval[1] <= 1


def test_wilson_matches_optional_external_reference() -> None:
    # The declared PyMatching extra supplies SciPy. It is not a separate
    # optional integration or a new dependency of the statistics example.
    pytest.importorskip("pymatching")
    from scipy.stats import binomtest

    for failures, shots in _CASES:
        expected = binomtest(failures, shots).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        assert baseline._wilson_interval(failures, shots) == pytest.approx(
            (expected.low, expected.high), abs=1e-14
        )


def test_no_failures_is_not_a_zero_probability_claim() -> None:
    lower, upper = baseline._wilson_interval(0, 100)
    assert lower == 0 and 0 < upper < 1
    lower, upper = baseline._wilson_interval(100, 100)
    assert 0 < lower < 1 and upper == 1
    with pytest.raises(ValueError, match="positive shots"):
        baseline._wilson_interval(0, 0)


def test_boundary_intervals_include_exact_zero_and_one_despite_roundoff() -> None:
    for shots in (4, 9, 15, 17, 21, 128, 1000, 1_000_000):
        assert baseline._wilson_interval(0, shots)[0] == 0.0
        assert baseline._wilson_interval(shots, shots)[1] == 1.0


@pytest.mark.parametrize(("failures", "shots"), [(-1, 10), (11, 10), (0, -1)])
def test_invalid_counts_have_no_interval(failures: int, shots: int) -> None:
    with pytest.raises(ValueError):
        baseline._wilson_interval(failures, shots)


@pytest.mark.parametrize(
    ("failures", "shots"), [(True, 10), (0, True), (0.0, 10), (0, 10.5)]
)
def test_non_integer_counts_are_not_coerced(failures: object, shots: object) -> None:
    with pytest.raises(TypeError):
        baseline._wilson_interval(failures, shots)


def test_any_observable_mismatch_counts_a_shot_once() -> None:
    predicted = torch.tensor([[0, 1], [1, 1], [0, 0]], dtype=torch.bool)
    observed = torch.zeros((3, 2), dtype=torch.int8)
    assert baseline._logical_failure_counts(predicted, observed) == (2, [1, 2])


@pytest.mark.parametrize(
    ("predicted", "observed"),
    [
        (torch.zeros(2), torch.zeros((2, 1))),
        (torch.zeros((2, 1)), torch.zeros((2, 2))),
        (torch.zeros((2, 0)), torch.zeros((2, 0))),
        (torch.tensor([[2]]), torch.tensor([[0]])),
        (torch.tensor([[0]]), torch.tensor([[float("nan")]])),
    ],
)
def test_malformed_logical_batches_are_not_counted(
    predicted: torch.Tensor, observed: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        baseline._logical_failure_counts(predicted, observed)


def _batch(shots: int, *, flipped: bool = False) -> DemSample:
    return DemSample(
        detectors=torch.zeros((shots, 8), dtype=torch.int8),
        observables=torch.full((shots, 1), int(flipped), dtype=torch.int8),
    )


def _decoder() -> Mock:
    decoder = Mock(spec=PyMatchingDecoder)
    decoder.decode_batch.side_effect = lambda events: torch.zeros(
        (events.shape[0], 1), dtype=torch.bool
    )
    return decoder


def test_tail_batch_is_in_the_fixed_denominator() -> None:
    sampler = Mock(spec=MemorySampler)
    sampler.sample.side_effect = [_batch(16), _batch(16), _batch(3, flipped=True)]
    decoder = _decoder()
    assert baseline._collect_counts(
        sampler, decoder, shots=35, batch_size=16, observables=1
    ) == (35, 3, [3])
    assert [call.kwargs["shots"] for call in sampler.sample.call_args_list] == [
        16,
        16,
        3,
    ]


@pytest.mark.parametrize(
    "bad_batch", [_batch(15), DemSample(torch.zeros((16, 8)), torch.zeros((16, 2)))]
)
def test_incomplete_or_wrong_width_batch_is_refused(bad_batch: DemSample) -> None:
    sampler = Mock(spec=MemorySampler)
    sampler.sample.return_value = bad_batch
    decoder = _decoder()
    with pytest.raises(ValueError, match="incomplete or malformed"):
        baseline._collect_counts(
            sampler, decoder, shots=16, batch_size=16, observables=1
        )
    decoder.decode_batch.assert_not_called()


def test_decoder_failure_preserves_error_and_records_completed_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sampler = Mock(spec=MemorySampler)
    sampler.sample.side_effect = [_batch(16), _batch(16), _batch(3)]
    decoder = _decoder()
    error = CapabilityError("unreachable syndrome")
    decoder.decode_batch.side_effect = [torch.zeros((16, 1), dtype=torch.bool), error]
    with caplog.at_level(logging.ERROR), pytest.raises(CapabilityError) as caught:
        baseline._collect_counts(
            sampler, decoder, shots=35, batch_size=16, observables=1
        )
    assert caught.value is error
    assert sampler.sample.call_count == decoder.decode_batch.call_count == 2
    assert "completed=16, batch_start=16, batch_shots=16" in caplog.text


def test_sampling_failure_does_not_invent_successful_rows() -> None:
    sampler = Mock(spec=MemorySampler)
    error = RuntimeError("sampling failed")
    sampler.sample.side_effect = error
    decoder = _decoder()
    with pytest.raises(RuntimeError) as caught:
        baseline._collect_counts(
            sampler, decoder, shots=35, batch_size=16, observables=1
        )
    assert caught.value is error
    decoder.decode_batch.assert_not_called()


@pytest.fixture
def workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, Mock]:
    sampler = MagicMock(spec=MemorySampler)
    sampler.__enter__.return_value = sampler
    sampler.sample.side_effect = lambda *, shots: _batch(shots)
    monkeypatch.setattr(baseline, "compile_memory_sampler", Mock(return_value=sampler))
    decoder = _decoder()
    monkeypatch.setattr(
        baseline.PyMatchingDecoder,
        "from_detector_error_model",
        Mock(return_value=decoder),
    )
    monkeypatch.setattr(baseline, "version", lambda name: "test-version")
    return sampler, decoder


def test_workflow_reports_metadata_and_closes_without_optional_sdks(
    workflow: tuple[MagicMock, Mock],
) -> None:
    sampler, _ = workflow
    report = baseline.run_baseline(shots=35, batch_size=16)
    assert report["requested_shots"] == report["shots"] == 35
    assert report["logical_failures"] == 0 and report["observable_failures"] == [0]
    assert report["logical_error_rate"] == 0
    assert report["confidence_interval"][1] > 0
    assert report["interval_method"] == "wilson" and report["confidence_level"] == 0.95
    assert report["stopping_rule"] == "fixed_shots"
    assert (
        report["sampling"] == "circuit" and report["rounds"] == report["distance"] == 3
    )
    assert (
        report["noise"]["data_flip"] == 0.02
        and report["noise"]["measurement_flip"] == 0.03
    )
    assert report["seed"] == 53 and report["seed_schedule"] == "continuous_stream"
    assert report["batch_size"] == 16 and report["versions"]["stim"] == "test-version"
    sampler.__exit__.assert_called_once_with(None, None, None)


def test_failed_workflow_exits_context_and_returns_no_partial_report(
    workflow: tuple[MagicMock, Mock],
) -> None:
    sampler, decoder = workflow
    error = CapabilityError("decoder execution failed")
    decoder.decode_batch.side_effect = error
    with pytest.raises(CapabilityError) as caught:
        baseline.run_baseline(shots=35, batch_size=16)
    assert caught.value is error
    sampler.__exit__.assert_called_once()
    error_type, error_value, traceback = sampler.__exit__.call_args.args
    assert error_type is CapabilityError and error_value is error
    assert traceback is not None


@pytest.mark.parametrize("basis", ["x", "z"])
def test_surface_workflow_uses_existing_statistics_without_optional_sdks(
    basis: str, workflow: tuple[MagicMock, Mock]
) -> None:
    sampler, _ = workflow
    report = baseline.run_surface_baseline(readout_basis=basis, shots=35, batch_size=16)
    assert report["shots"] == report["requested_shots"] == 35
    assert report["readout_basis"] == basis
    assert report["logical_support"] == list(range(9 if basis == "x" else 3))
    assert report["noise"]["phase_flip"] == 0.007
    assert report["confidence_interval"] == baseline._wilson_interval(0, 35)
    sampler.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize(
    "arguments",
    [{"distance": 4}, {"distance": True}, {"readout_basis": "y"}, {"shots": 0}],
)
def test_invalid_surface_requests_are_refused_before_preparation(
    arguments: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_sampler = Mock()
    monkeypatch.setattr(baseline, "compile_memory_sampler", compile_sampler)
    with pytest.raises((TypeError, ValueError)):
        baseline.run_surface_baseline(**arguments)
    compile_sampler.assert_not_called()


@pytest.mark.parametrize(
    ("shots", "batch_size"), [(0, 16), (35, 0), (True, 16), (35, 1.5)]
)
def test_invalid_requested_work_is_refused_before_preparation(
    shots: object, batch_size: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_sampler = Mock()
    monkeypatch.setattr(baseline, "compile_memory_sampler", compile_sampler)
    with pytest.raises((TypeError, ValueError)):
        baseline.run_baseline(shots=shots, batch_size=batch_size)
    compile_sampler.assert_not_called()
