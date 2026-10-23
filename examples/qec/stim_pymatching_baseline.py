"""Run CPU circuit sampling and batched logical prediction.

Install the stim and pymatching extras, then run with PYTHONPATH=. from the
checkout. This example is not a performance benchmark. Memory is bounded by
batch_size; sampler compilation is still repeated per chunk.
"""

from __future__ import annotations

import json

from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    PyMatchingDecoder,
    RepetitionCode,
    build_memory_circuit,
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
    decoder = PyMatchingDecoder.from_detector_error_model(
        model.merge_duplicate_mechanisms()
    )
    failures = 0
    completed = 0
    for batch_index, start in enumerate(range(0, shots, batch_size)):
        sample = sample_memory_circuit(
            memory,
            noise=noise,
            shots=min(batch_size, shots - start),
            seed=53 + batch_index,
        )
        predicted = decoder.decode_batch(sample.detectors)
        failures += int((predicted != sample.observables.bool()).any(dim=1).sum())
        completed += sample.shots
    return {
        "sampling": "circuit",
        "decoder": "pymatching_batch",
        "shots": completed,
        "logical_failures": failures,
        "detectors": model.num_detectors,
        "observables": model.num_observables,
    }


if __name__ == "__main__":
    print(json.dumps(run_baseline(), sort_keys=True))
