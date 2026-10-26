"""Private fixed-grid temperature fitting and validation-only threshold choice."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from flagquantum.qec.decoders._ai.assessment import PairedCounts, assess
from flagquantum.qec.decoders._ai.diagnostics import checked_scores, diagnose
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders.registry import DetectorErrorModelDecoder


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    raw_mean_bce: float
    calibrated_mean_bce: float
    raw_brier: float
    calibrated_brier: float


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperatures: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> TemperatureFit:
    scores, targets = checked_scores(logits, labels)
    if not temperatures or len(set(temperatures)) != len(temperatures):
        raise ValueError("temperature grid must be nonempty and unique")
    for value in temperatures:
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("temperatures must be finite and positive")
    raw = diagnose(scores, targets)
    candidates = [(value, diagnose(scores / value, targets)) for value in temperatures]
    temperature, selected = min(
        candidates,
        key=lambda item: (item[1].mean_bce, abs(math.log(item[0])), item[0]),
    )
    return TemperatureFit(
        temperature,
        raw.mean_bce,
        selected.mean_bce,
        raw.brier_score,
        selected.brier_score,
    )


@dataclass(frozen=True)
class ThresholdTrial:
    threshold: float
    counts: PairedCounts


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    trials: tuple[ThresholdTrial, ...]


def _select_threshold_trials(trials: tuple[ThresholdTrial, ...]) -> ThresholdSelection:
    """Share the conservative tie rule with saved physical-validation trials."""
    thresholds = tuple(trial.threshold for trial in trials)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold grid must be nonempty and unique")
    for value in thresholds:
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("thresholds must be finite and in (0, 1]")
    selected = min(
        trials, key=lambda trial: (trial.counts.hybrid_failures, -trial.threshold)
    )
    return ThresholdSelection(selected.threshold, trials)


def select_threshold(
    model: Predecoder,
    backend: DetectorErrorModelDecoder,
    *,
    shots: int,
    seed: int,
    temperature: float = 1.0,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.9, 0.99, 1.0),
) -> ThresholdSelection:
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold grid must be nonempty and unique")
    for value in thresholds:
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("thresholds must be finite and in (0, 1]")
    trials = tuple(
        ThresholdTrial(
            threshold,
            assess(
                model,
                backend,
                shots=shots,
                seed=seed,
                threshold=threshold,
                temperature=temperature,
            ),
        )
        for threshold in thresholds
    )
    return _select_threshold_trials(trials)
