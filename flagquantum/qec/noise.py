"""Circuit-location noise profiles and finite-shot QEC sweeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable

from ..noise import NoiseModel, ReadoutError, bit_flip_channel
from .repetition import run_repetition_memory_experiment
from .types import NoiseSweepPoint


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real probability")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return normalized


def _symmetric_readout_error(probability: float) -> ReadoutError:
    return ReadoutError(
        ((1.0 - probability, probability), (probability, 1.0 - probability))
    )


@dataclass(frozen=True)
class RepetitionNoiseProfile:
    """Bounded circuit-location noise configuration for the reference code."""

    data_bit_flip_probability: float = 0.0
    syndrome_readout_error_probability: float = 0.0
    final_readout_error_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "data_bit_flip_probability",
            "syndrome_readout_error_probability",
            "final_readout_error_probability",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name=name),
            )

    def to_noise_model(self) -> NoiseModel:
        """Map QEC locations onto the backend-neutral NoiseModel authority."""

        model = NoiseModel()
        if self.data_bit_flip_probability:
            channel = bit_flip_channel(self.data_bit_flip_probability)
            for wire in range(3):
                model.add("cx", channel, wires=wire)
        if self.syndrome_readout_error_probability:
            error = _symmetric_readout_error(self.syndrome_readout_error_probability)
            model.add_readout((3, 4), error)
        if self.final_readout_error_probability:
            error = _symmetric_readout_error(self.final_readout_error_probability)
            model.add_readout((0, 1, 2), error)
        return model


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


__all__ = (
    "RepetitionNoiseProfile",
    "run_repetition_memory_noise_sweep",
)
