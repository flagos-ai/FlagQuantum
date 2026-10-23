"""Independent layout and decoder evidence for the pre-batch vertical path."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch

pytest.importorskip("stim")
pytest.importorskip("pymatching")

from flagquantum.qec import (
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
    decoder_context_from_memory_circuit,
    get_decoder,
    sample_memory_circuit,
    sample_memory_measurements,
)

pytestmark = pytest.mark.integration


def test_raw_measurement_parities_match_sampled_events_and_logicals() -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    noise = PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.03)
    context = decoder_context_from_memory_circuit(memory, noise=noise)
    inputs = context.full_component()
    raw = sample_memory_measurements(memory, noise=noise, shots=64, seed=53)
    sample = sample_memory_circuit(memory, noise=noise, shots=64, seed=53)
    bits = raw.outcomes.to(torch.int64)
    detector_map = inputs.measurement_to_detectors.dense().to(torch.int64)
    observable_map = inputs.measurement_to_observables.dense().to(torch.int64)
    assert torch.equal((bits @ detector_map.T) % 2, sample.detectors)
    assert torch.equal((bits @ observable_map.T) % 2, sample.observables)


def test_matching_predictions_agree_with_upstream_batch_on_same_graph() -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    noise = PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.03)
    inputs = decoder_context_from_memory_circuit(memory, noise=noise).full_component()
    decoder = get_decoder("pymatching", inputs.dem.merge_duplicate_mechanisms())
    sample = sample_memory_circuit(memory, noise=noise, shots=64, seed=71)
    # Private inspection stays in this test: the example never exposes SDK objects.
    upstream = decoder._matching.decode_batch(sample.detectors.numpy())
    predictions = torch.zeros_like(sample.observables)
    for row, events in enumerate(sample.detectors):
        indices = tuple(int(index) for index in events.nonzero().flatten())
        for observable in decoder.decode(indices).observables:
            predictions[row, observable] = 1
    assert torch.equal(predictions, torch.as_tensor(upstream, dtype=predictions.dtype))


def test_example_counts_tail_batch_without_discarding_shots() -> None:
    path = Path(__file__).parents[2] / "examples/qec/stim_pymatching_baseline.py"
    example = runpy.run_path(str(path))
    report = example["run_baseline"](shots=35, batch_size=16)
    assert report["shots"] == 35
    assert report["sampling"] == "circuit"
    assert 0 <= report["logical_failures"] <= 35
    assert report["observables"] == 1
