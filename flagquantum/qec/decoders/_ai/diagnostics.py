"""Private label-sparsity and loss diagnostics for CPU mechanism scores."""

from __future__ import annotations

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
