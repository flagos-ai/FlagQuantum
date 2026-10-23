"""Finite-shot QEC sweeps over the existing repetition-memory executor."""

from __future__ import annotations

from collections.abc import Iterable

from ..noise import RepetitionNoiseProfile, _probability
from ..repetition import run_repetition_memory_experiment
from ..types import NoiseSweepPoint


def run_repetition_memory_noise_sweep(
    probabilities: Iterable[float],
    *,
    rounds: int = 3,
    shots: int = 1024,
    seed: int | None = 0,
    strategy: str = "auto",
    feedback_mode: str = "compiled_lookup",
    syndrome_readout_error_probability: float = 0.0,
    final_readout_error_probability: float = 0.0,
) -> tuple[NoiseSweepPoint, ...]:
    """Measure finite-shot logical outcomes without a suppression claim."""

    values = tuple(
        _probability(value, name="data_bit_flip_probability") for value in probabilities
    )
    if not values:
        raise ValueError("noise sweep requires at least one probability")
    points = []
    for index, probability in enumerate(values):
        profile = RepetitionNoiseProfile(
            data_bit_flip_probability=probability,
            syndrome_readout_error_probability=syndrome_readout_error_probability,
            final_readout_error_probability=final_readout_error_probability,
        )
        result = run_repetition_memory_experiment(
            rounds=rounds,
            shots=shots,
            seed=None if seed is None else seed + index,
            strategy=strategy,
            feedback_mode=feedback_mode,
            noise_model=profile.to_noise_model(),
        )
        points.append(
            NoiseSweepPoint(
                probability=probability,
                shots=result.shot_count,
                logical_failures=result.logical_failures,
                logical_error_rate=result.logical_error_rate,
                bit_flip_events=result.bit_flip_events,
                readout_errors=result.readout_errors,
            )
        )
    return tuple(points)


# Preserve the public function's historical pickle and reflection path.
run_repetition_memory_noise_sweep.__module__ = "flagquantum.qec.noise"
