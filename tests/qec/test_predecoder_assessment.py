"""Evidence gates must reject weak samples and keep baseline inference neural-free."""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.qec import get_decoder
from flagquantum.qec.decoders._assessment import (
    PairedCounts,
    assess,
    baseline_decode,
    block_timings,
    paired_counts,
    paired_p_value,
    rate_interval,
    summarize,
)
from flagquantum.qec.decoders._neural import Predecoder
from tests.qec.test_neural_training_data import _experiment

pytestmark = pytest.mark.integration


def test_paired_multi_observable_counts() -> None:
    truth = torch.zeros(4, 2)
    first = torch.tensor([[0, 1], [0, 0], [1, 0], [0, 0]])
    second = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 0]])
    assert paired_counts(truth, first, second) == PairedCounts(4, 2, 2, 1, 1)
    with pytest.raises(ValueError, match="binary"):
        paired_counts(truth, first * 2, second)


def test_zero_errors_are_not_zero_risk_or_evidence_of_gain() -> None:
    lower, upper = rate_interval(0, 100, 0.05)
    assert lower == 0
    assert upper == pytest.approx(1 - 0.025**0.01)
    report = summarize(PairedCounts(100, 0, 0, 0, 0))
    assert report["ratio_upper"] is None
    assert report["quality_gate"] == "inconclusive"
    assert paired_p_value(0, 0) == 1
    assert paired_p_value(10, 0) == pytest.approx(2 / 1024)
    assert paired_p_value(5, 5) == pytest.approx(1)


def test_quality_gate_needs_precision_not_just_lower_point_rate() -> None:
    assert summarize(PairedCounts(100, 2, 1, 1, 0))["quality_gate"] == "inconclusive"
    assert (
        summarize(PairedCounts(1000000, 200000, 100000, 100000, 0))["quality_gate"]
        == "supported_10_percent_reduction"
    )
    with pytest.raises(ValueError):
        rate_interval(1, 0, 0.05)
    with pytest.raises(ValueError):
        rate_interval(1, 10, math.nan)


def test_baseline_never_calls_the_model_and_evaluation_replays() -> None:
    _, _, dem = _experiment("z", 1, 0.02)
    model = Predecoder(dem)
    backend = get_decoder("minimum_weight_matching", dem)
    sample = model.sample(16, torch.Generator().manual_seed(17)).sample
    # Poison the neural path; pure baseline must still execute successfully.
    model.network[-1].bias.data.fill_(math.nan)
    result = baseline_decode(sample.detectors, backend, dem.num_observables)
    assert result.shape == sample.observables.shape
    model.network[-1].bias.data.zero_()
    first = assess(model, backend, shots=31, seed=23, threshold=1.0, batch_size=8)
    second = assess(model, backend, shots=31, seed=23, threshold=1.0, batch_size=8)
    assert first == second
    assert first.wins == first.losses == 0
    timing = block_timings(model, backend, seed=23, threshold=1.0, repetitions=4)
    assert timing["scope"] == "cpu_batch_one_offline_block"
    assert timing["repetitions"] == 4
