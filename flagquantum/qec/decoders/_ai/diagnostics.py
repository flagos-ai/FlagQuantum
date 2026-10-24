"""Private CPU mechanism-score, correction and representation diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as functional


def checked_scores(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("diagnostics require CPU tensors")
    if logits.ndim != 2 or logits.numel() == 0 or labels.shape != logits.shape:
        raise ValueError("requires equally shaped nonempty shot/mechanism arrays")
    if not logits.is_floating_point() or labels.is_complex():
        raise ValueError("logits must be real floating-point and labels real binary")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("mechanism logits must be finite")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("mechanism labels must be binary")
    return logits.detach().double(), labels.detach().double()


@dataclass(frozen=True)
class LossDiagnostics:
    total_labels: int
    positive_labels: int
    negative_labels: int
    positive_bce_sum: float
    negative_bce_sum: float
    predicted_positive_labels: int
    mean_probability: float
    brier_score: float

    @property
    def mean_bce(self) -> float:
        return (self.positive_bce_sum + self.negative_bce_sum) / self.total_labels


def diagnose(logits: torch.Tensor, labels: torch.Tensor) -> LossDiagnostics:
    scores, targets = checked_scores(logits, labels)
    losses = functional.binary_cross_entropy_with_logits(
        scores, targets, reduction="none"
    )
    positive = targets == 1
    probabilities = scores.sigmoid()
    count = int(positive.sum())
    return LossDiagnostics(
        total_labels=targets.numel(),
        positive_labels=count,
        negative_labels=targets.numel() - count,
        positive_bce_sum=float(losses[positive].sum()),
        negative_bce_sum=float(losses[~positive].sum()),
        predicted_positive_labels=int((probabilities >= 0.5).sum()),
        mean_probability=float(probabilities.mean()),
        brier_score=float((probabilities - targets).square().mean()),
    )


@dataclass(frozen=True)
class EquivalenceDiagnostics:
    shots: int
    latent_equal: int
    detector_equal: int
    detector_and_logical_equal: int
    detector_equal_logical_different: int
    latent_mismatch_bits: int


def _binary_matrix(value: torch.Tensor) -> None:
    if value.device.type != "cpu":
        raise ValueError("equivalence diagnostics require CPU tensors")
    if value.ndim != 2 or value.is_complex():
        raise ValueError("requires real binary matrices")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError("requires real binary matrices")


def diagnose_equivalence(
    h: torch.Tensor,
    logical: torch.Tensor,
    target: torch.Tensor,
    candidate: torch.Tensor,
) -> EquivalenceDiagnostics:
    """Compare latent corrections through canonical complete H/L projections.

    This describes mechanism supervision, not final global-decoder success.
    Zero logical rows are valid; empty shot/mechanism axes are refused.
    """
    for value in (h, logical, target, candidate):
        _binary_matrix(value)
    if (
        h.numel() == 0
        or target.numel() == 0
        or target.shape != candidate.shape
        or target.shape[1] != h.shape[1]
        or logical.shape[1] != h.shape[1]
    ):
        raise ValueError("incompatible nonempty shot/mechanism/projection shapes")
    difference = (target != candidate).double()
    detector_same = ((difference @ h.double().T).remainder(2) == 0).all(dim=1)
    logical_same = ((difference @ logical.double().T).remainder(2) == 0).all(dim=1)
    return EquivalenceDiagnostics(
        shots=target.shape[0],
        latent_equal=int((difference == 0).all(dim=1).sum()),
        detector_equal=int(detector_same.sum()),
        detector_and_logical_equal=int((detector_same & logical_same).sum()),
        detector_equal_logical_different=int((detector_same & ~logical_same).sum()),
        latent_mismatch_bits=int(difference.sum()),
    )


@dataclass(frozen=True)
class FeatureGroup:
    features: tuple[float, float, float]
    observations: int
    positive_labels: int
    logit_sum: float
    probability_sum: float
    minimum_logit: float
    maximum_logit: float

    @property
    def negative_labels(self) -> int:
        return self.observations - self.positive_labels

    @property
    def positive_fraction(self) -> float:
        return self.positive_labels / self.observations

    @property
    def mean_logit(self) -> float:
        return self.logit_sum / self.observations

    @property
    def mean_probability(self) -> float:
        return self.probability_sum / self.observations

    @property
    def empirical_entropy_nats(self) -> float:
        p = self.positive_fraction
        return -(p * math.log(p) + (1 - p) * math.log1p(-p)) if 0 < p < 1 else 0.0


@dataclass(frozen=True)
class FeatureAmbiguity:
    groups: tuple[FeatureGroup, ...]
    observations: int
    singleton_groups: int
    mixed_label_groups: int
    mixed_label_observations: int
    empirical_conditional_bce: float


class FeatureTable:
    """Stream exact three-feature MLP groups without retaining all model-shots.

    Counts and the empirical group-frequency optimum describe this sample only.
    They do not estimate a population Bayes bound or logical decoding accuracy.
    Memory grows with distinct exact triples, not with the number of shots.
    """

    def __init__(self) -> None:
        self._groups: dict[tuple[float, float, float], FeatureGroup] = {}

    def update(
        self, features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> None:
        scores, targets = checked_scores(logits, labels)
        if features.device.type != "cpu":
            raise ValueError("feature diagnostics require CPU tensors")
        if (
            features.shape != (*scores.shape, 3)
            or not features.is_floating_point()
            or not bool(torch.isfinite(features).all())
        ):
            raise ValueError(
                "requires finite exact shot/mechanism/three-feature inputs"
            )
        triples, inverse = torch.unique(
            features.detach().reshape(-1, 3), dim=0, return_inverse=True
        )
        n = triples.shape[0]
        counts = torch.bincount(inverse, minlength=n)
        positives = torch.zeros(n, dtype=torch.float64).scatter_add_(
            0, inverse, targets.flatten()
        )
        sums = torch.zeros(n, dtype=torch.float64).scatter_add_(
            0, inverse, scores.flatten()
        )
        probabilities = torch.zeros(n, dtype=torch.float64).scatter_add_(
            0, inverse, scores.sigmoid().flatten()
        )
        minima = torch.full((n,), math.inf, dtype=torch.float64).scatter_reduce_(
            0, inverse, scores.flatten(), reduce="amin"
        )
        maxima = torch.full((n,), -math.inf, dtype=torch.float64).scatter_reduce_(
            0, inverse, scores.flatten(), reduce="amax"
        )
        for i, triple in enumerate(triples):
            key = (float(triple[0]), float(triple[1]), float(triple[2]))
            old = self._groups.get(key)
            self._groups[key] = FeatureGroup(
                features=key,
                observations=int(counts[i]) + (old.observations if old else 0),
                positive_labels=int(positives[i]) + (old.positive_labels if old else 0),
                logit_sum=float(sums[i]) + (old.logit_sum if old else 0.0),
                probability_sum=float(probabilities[i])
                + (old.probability_sum if old else 0.0),
                minimum_logit=min(
                    float(minima[i]), old.minimum_logit if old else math.inf
                ),
                maximum_logit=max(
                    float(maxima[i]), old.maximum_logit if old else -math.inf
                ),
            )

    def summarize(self) -> FeatureAmbiguity:
        if not self._groups:
            raise ValueError("feature diagnostics require at least one observation")
        groups = tuple(self._groups[key] for key in sorted(self._groups))
        total = sum(g.observations for g in groups)
        mixed = tuple(g for g in groups if 0 < g.positive_labels < g.observations)
        return FeatureAmbiguity(
            groups=groups,
            observations=total,
            singleton_groups=sum(g.observations == 1 for g in groups),
            mixed_label_groups=len(mixed),
            mixed_label_observations=sum(g.observations for g in mixed),
            empirical_conditional_bce=sum(
                g.observations * g.empirical_entropy_nats for g in groups
            )
            / total,
        )
