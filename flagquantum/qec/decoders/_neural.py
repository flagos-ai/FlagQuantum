"""CPU research pre-decoder; private, independent graphlike DEMs only.

The shared network scores mechanisms from incident detector bits and their prior.
This bounded M1 reference makes no accuracy, scaling or latency claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from flagquantum.qec.decoders.registry import DetectorErrorModelDecoder
from flagquantum.qec.dem.model import DemSample, DetectorErrorModel


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TrainingBatch:
    """Mechanisms are supervision only; inference consumes detectors alone."""

    sample: DemSample
    mechanisms: torch.Tensor


class Predecoder(nn.Module):
    """Shared local mechanism scorer with explicit DEM-bound buffers."""

    h: torch.Tensor
    logical: torch.Tensor
    prior: torch.Tensor

    def __init__(self, dem: DetectorErrorModel) -> None:
        super().__init__()
        if not dem.errors or dem.exclusive_groups():
            raise ValueError("requires nonempty independent mechanisms")
        dem.require_unique_mechanisms()
        if any(len(e.detectors) not in (1, 2) for e in dem.errors):
            raise ValueError("requires graphlike mechanisms with visible detectors")
        if any(not 0 < e.probability < 0.5 for e in dem.errors):
            raise ValueError("requires mechanism probabilities in (0, 0.5)")
        self.dem_text = dem.to_stim_text()
        self.register_buffer("h", dem.detector_error_matrix().float())
        self.register_buffer("logical", dem.observables_flips_matrix().float())
        self.register_buffer("prior", torch.tensor([e.probability for e in dem.errors]))
        self.network = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, detectors: torch.Tensor) -> torch.Tensor:
        """Return one logit per mechanism, from syndrome bits only."""
        if detectors.ndim != 2 or detectors.shape[1] != self.h.shape[0]:
            raise ValueError("detector batch has incompatible shape")
        if not bool(((detectors == 0) | (detectors == 1)).all()):
            raise ValueError("detectors must be binary")
        active = detectors.float() @ self.h
        degree = self.h.sum(0).expand_as(active)
        prior = torch.logit(self.prior).expand_as(active)
        logits: torch.Tensor = self.network(
            torch.stack((active, degree, prior), dim=-1)
        )
        return logits.squeeze(-1)

    def sample(self, shots: int, generator: torch.Generator) -> TrainingBatch:
        """Draw a bounded training batch, preserving the latent mechanism vector."""
        _positive(shots, "shots")
        fired = (
            torch.rand((shots, self.prior.numel()), generator=generator) < self.prior
        ).float()
        return TrainingBatch(
            DemSample(
                (fired @ self.h.T).remainder(2).to(torch.int8),
                (fired @ self.logical.T).remainder(2).to(torch.int8),
            ),
            fired,
        )

    @torch.no_grad()
    def decode(
        self,
        detectors: torch.Tensor,
        backend: DetectorErrorModelDecoder,
        *,
        threshold: float = 0.9,
        enabled: bool = True,
    ) -> torch.Tensor:
        """Combine the local correction's logical contribution with residual decode."""
        if isinstance(threshold, bool) or not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        logits = self(detectors)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite predecoder output")
        correction = (
            (logits.sigmoid() >= threshold)
            if enabled
            else torch.zeros_like(logits, dtype=torch.bool)
        ).float()
        residual = (detectors.float() + correction @ self.h.T).remainder(2)
        prediction = (correction @ self.logical.T).remainder(2).to(torch.int8)
        for i, row in enumerate(residual):
            result = backend.decode(row.nonzero().flatten().tolist())
            for observable in result.observables:
                prediction[i, observable] ^= 1
        return prediction


class Trainer:
    """CPU Adam training with exact batch-stream and optimizer resume."""

    def __init__(
        self, model: Predecoder, *, seed: int = 0, learning_rate: float = 0.001
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.generator = torch.Generator().manual_seed(seed)
        self.step = 0

    def train(self, steps: int, *, batch_size: int = 128) -> list[float]:
        _positive(steps, "steps")
        _positive(batch_size, "batch_size")
        losses: list[float] = []
        self.model.train()
        for _ in range(steps):
            batch = self.model.sample(batch_size, self.generator)
            loss = nn.functional.binary_cross_entropy_with_logits(
                self.model(batch.sample.detectors), batch.mechanisms
            )
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            self.optimizer.zero_grad()
            torch.autograd.backward(loss)
            self.optimizer.step()
            self.step += 1
            losses.append(float(loss.detach()))
        return losses

    def save(self, path: Path) -> None:
        torch.save(
            {
                "version": 1,
                "dem": self.model.dem_text,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "generator": self.generator.get_state(),
                "step": self.step,
            },
            path,
        )

    def restore(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["version"] != 1 or state["dem"] != self.model.dem_text:
            raise ValueError("checkpoint version or DEM does not match")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.step = state["step"]


def evaluate(
    model: Predecoder,
    backend: DetectorErrorModelDecoder,
    *,
    shots: int,
    seed: int,
    batch_size: int = 128,
    threshold: float = 0.9,
) -> dict[str, int | float]:
    """Paired held-out logical failure counts; never equate zero failures with zero risk."""
    _positive(shots, "shots")
    _positive(batch_size, "batch_size")
    generator = torch.Generator().manual_seed(seed)
    baseline = hybrid = 0
    model.eval()
    for start in range(0, shots, batch_size):
        batch = model.sample(min(batch_size, shots - start), generator)
        truth = batch.sample.observables
        baseline += int(
            (model.decode(batch.sample.detectors, backend, enabled=False) != truth)
            .any(1)
            .sum()
        )
        hybrid += int(
            (
                model.decode(batch.sample.detectors, backend, threshold=threshold)
                != truth
            )
            .any(1)
            .sum()
        )
    return {
        "shots": shots,
        "baseline_failures": baseline,
        "hybrid_failures": hybrid,
        "baseline_ler": baseline / shots,
        "hybrid_ler": hybrid / shots,
    }
