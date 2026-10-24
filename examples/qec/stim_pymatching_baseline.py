"""Evaluate fixed-shot CPU circuit samples with batched logical prediction.

Install the stim and pymatching extras, then run with PYTHONPATH=. from the
checkout. One compiled sampler serves all chunks; sample storage is bounded by
batch_size. The Wilson interval estimates the probability that any logical bit
is decoded incorrectly in one complete memory experiment, at a fixed shot count.
"""

from __future__ import annotations

import json
import logging
import math
import platform
from importlib.metadata import version
from statistics import NormalDist
from typing import TypedDict

import torch

from flagquantum import __version__
from flagquantum.qec import (
    DetectorErrorModel,
    MemorySampler,
    PhenomenologicalNoise,
    PyMatchingDecoder,
    RepetitionCode,
    build_memory_circuit,
    compile_memory_sampler,
)

_LOGGER = logging.getLogger(__name__)
_CONFIDENCE_LEVEL = 0.95
_NORMAL_QUANTILE = NormalDist().inv_cdf((1 + _CONFIDENCE_LEVEL) / 2)


class _NoiseSettings(TypedDict):
    data_flip: float
    phase_flip: float
    both_flip: float
    measurement_flip: float


class _SoftwareVersions(TypedDict):
    flagquantum: str
    torch: str
    stim: str
    pymatching: str
    python: str


class _BaselineReport(TypedDict):
    sampling: str
    decoder: str
    requested_shots: int
    shots: int
    logical_failures: int
    observable_failures: list[int]
    logical_error_rate: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    interval_method: str
    stopping_rule: str
    detectors: int
    observables: int
    code: str
    distance: int
    rounds: int
    noise: _NoiseSettings
    seed: int
    seed_schedule: str
    batch_size: int
    versions: _SoftwareVersions
    platform: str


def _wilson_interval(failures: int, shots: int) -> tuple[float, float]:
    """Two-sided 95% score interval without continuity correction.

    Formula: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
    Zero completed shots have no defined estimate or interval.
    """
    for name, value in (("failures", failures), ("shots", shots)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if shots <= 0 or not 0 <= failures <= shots:
        raise ValueError(
            "an interval requires positive shots and 0 <= failures <= shots"
        )
    proportion = failures / shots
    squared_quantile = _NORMAL_QUANTILE**2
    denominator = 1 + squared_quantile / shots
    center = (proportion + squared_quantile / (2 * shots)) / denominator
    half_width = (
        _NORMAL_QUANTILE
        * math.sqrt(
            proportion * (1 - proportion) / shots + squared_quantile / (4 * shots**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _logical_failure_counts(
    predicted: torch.Tensor, observed: torch.Tensor
) -> tuple[int, list[int]]:
    if predicted.ndim != 2 or predicted.shape != observed.shape:
        raise ValueError(
            "predicted and observed logical bits must have the same (shots, observables) shape"
        )
    if predicted.shape[1] == 0:
        raise ValueError("logical evaluation requires at least one observable")
    for bits in (predicted, observed):
        if not bool(((bits == 0) | (bits == 1)).all()):
            raise ValueError("logical predictions and observations must be binary")
    mismatches = predicted != observed
    return int(mismatches.any(dim=1).sum()), [
        int(count) for count in mismatches.sum(dim=0)
    ]


def _collect_counts(
    sampler: MemorySampler,
    decoder: PyMatchingDecoder,
    *,
    shots: int,
    batch_size: int,
    observables: int,
) -> tuple[int, int, list[int]]:
    completed = 0
    failures = 0
    observable_failures = [0] * observables
    for start in range(0, shots, batch_size):
        batch_shots = min(batch_size, shots - start)
        try:
            sample = sampler.sample(shots=batch_shots)
            if sample.shots != batch_shots or sample.observables.shape != (
                batch_shots,
                observables,
            ):
                raise ValueError(
                    "sampler returned an incomplete or malformed logical batch"
                )
            predicted = decoder.decode_batch(sample.detectors)
            batch_failures, per_observable = _logical_failure_counts(
                predicted, sample.observables
            )
        except Exception:
            # This workflow boundary records context and preserves the original
            # sampling/decoder error. An aborted evaluation returns no report.
            _LOGGER.error(
                "Fixed-shot evaluation failed: completed=%d, batch_start=%d, batch_shots=%d",
                completed,
                start,
                batch_shots,
            )
            raise
        failures += batch_failures
        observable_failures = [
            total + count
            for total, count in zip(observable_failures, per_observable, strict=True)
        ]
        completed += sample.shots
    return completed, failures, observable_failures


def run_baseline(*, shots: int = 128, batch_size: int = 32) -> _BaselineReport:
    """Report fixed-shot failures and uncertainty from the original circuit law.

    ``shots`` in the result is completed shots; every successful result completes
    ``requested_shots``. Sampling or decoding failure propagates with batch
    context logged and no partial result. The interval is shot-level, not a
    per-observable or per-round interval, and is not valid for adaptive stopping.
    """
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
    seed = 53
    with compile_memory_sampler(memory, noise=noise, seed=seed) as sampler:
        completed, failures, observable_failures = _collect_counts(
            sampler,
            decoder,
            shots=shots,
            batch_size=batch_size,
            observables=model.num_observables,
        )
    return {
        "sampling": "circuit",
        "decoder": "pymatching_batch",
        "requested_shots": shots,
        "shots": completed,
        "logical_failures": failures,
        "observable_failures": observable_failures,
        "logical_error_rate": failures / completed,
        "confidence_interval": _wilson_interval(failures, completed),
        "confidence_level": _CONFIDENCE_LEVEL,
        "interval_method": "wilson",
        "stopping_rule": "fixed_shots",
        "detectors": model.num_detectors,
        "observables": model.num_observables,
        "code": type(memory.code).__name__,
        "distance": memory.code.distance,
        "rounds": memory.rounds,
        "noise": {
            "data_flip": noise.data_flip,
            "phase_flip": noise.phase_flip,
            "both_flip": noise.both_flip,
            "measurement_flip": noise.measurement_flip,
        },
        "seed": seed,
        "seed_schedule": "continuous_stream",
        "batch_size": batch_size,
        "versions": {
            "flagquantum": __version__,
            "torch": torch.__version__,
            "stim": version("stim"),
            "pymatching": version("pymatching"),
            "python": platform.python_version(),
        },
        "platform": platform.platform(),
    }


if __name__ == "__main__":
    print(json.dumps(run_baseline(), sort_keys=True))
