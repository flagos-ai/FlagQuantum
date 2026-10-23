"""Run the existing CPU circuit-sampling and single-shot matching baseline.

Install the stim and pymatching extras, then run with PYTHONPATH=. from the
checkout. This baseline intentionally uses existing APIs; it is not the proposed
batch decoder or a performance benchmark. Memory is bounded by batch_size.
"""

from __future__ import annotations

import json

import torch

from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
    get_decoder,
    sample_memory_circuit,
)


def run_baseline(*, shots: int = 128, batch_size: int = 32) -> dict[str, int | str]:
    """Count logical failures without replacing the original sampling model."""
    for name, value in (("shots", shots), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    noise = PhenomenologicalNoise(data_flip=0.02, measurement_flip=0.03)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    decoder = get_decoder("pymatching", model.merge_duplicate_mechanisms())
    failures = 0
    completed = 0
    for batch_index, start in enumerate(range(0, shots, batch_size)):
        sample = sample_memory_circuit(
            memory,
            noise=noise,
            shots=min(batch_size, shots - start),
            seed=53 + batch_index,
        )
        for events, actual in zip(sample.detectors, sample.observables, strict=True):
            indices = tuple(int(index) for index in events.nonzero().flatten())
            result = decoder.decode(indices)
            predicted = torch.zeros_like(actual)
            for index in result.observables:
                predicted[index] = 1
            failures += int(not torch.equal(predicted, actual))
            completed += 1
    return {
        "sampling": "circuit",
        "decoder": "pymatching_single_shot",
        "shots": completed,
        "logical_failures": failures,
        "detectors": model.num_detectors,
        "observables": model.num_observables,
    }


if __name__ == "__main__":
    print(json.dumps(run_baseline(), sort_keys=True))
