"""Torch research pre-decoder; private, independent graphlike DEMs only.

The shared network scores mechanisms from incident detector bits and their prior.
This bounded M1 reference makes no accuracy, scaling or latency claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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

    def __init__(
        self, dem: DetectorErrorModel, *, allow_hyperedges: bool = False
    ) -> None:
        super().__init__()
        if not isinstance(allow_hyperedges, bool):
            raise ValueError("allow_hyperedges must be boolean")
        if not dem.errors or dem.exclusive_groups():
            raise ValueError("requires nonempty independent mechanisms")
        dem.require_unique_mechanisms()
        if any(not e.detectors for e in dem.errors):
            raise ValueError("requires mechanisms with visible detectors")
        if not allow_hyperedges and any(len(e.detectors) > 2 for e in dem.errors):
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
        logits: torch.Tensor = self.network(self._features(detectors))
        return logits.squeeze(-1)

    def _features(self, detectors: torch.Tensor) -> torch.Tensor:
        """The exact MLP inputs, shared with representation diagnostics."""
        if detectors.ndim != 2 or detectors.shape[1] != self.h.shape[0]:
            raise ValueError("detector batch has incompatible shape")
        if not bool(((detectors == 0) | (detectors == 1)).all()):
            raise ValueError("detectors must be binary")
        active = detectors.float() @ self.h
        degree = self.h.sum(0).expand_as(active)
        prior = torch.logit(self.prior).expand_as(active)
        return torch.stack((active, degree, prior), dim=-1)

    def sample(self, shots: int, generator: torch.Generator) -> TrainingBatch:
        """Draw a bounded training batch, preserving the latent mechanism vector."""
        _positive(shots, "shots")
        fired = (
            torch.rand(
                (shots, self.prior.numel()),
                generator=generator,
                device=self.prior.device,
            )
            < self.prior
        ).float()
        return TrainingBatch(
            DemSample(
                (fired @ self.h.T).remainder(2).to(torch.int8),
                (fired @ self.logical.T).remainder(2).to(torch.int8),
            ),
            fired,
        )

    @torch.no_grad()
    def correct(
        self,
        detectors: torch.Tensor,
        *,
        threshold: float = 0.9,
        enabled: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual detector bits and the local logical frame before decoding.

        Hyperedges retain their complete detector signature; this operation does
        not decompose or independently sample their individual detector flips.
        """
        if isinstance(threshold, bool) or not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if (
            isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive")
        logits = self(detectors)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite predecoder output")
        if temperature != 1.0:
            logits = logits / temperature
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite calibrated logits")
        correction = (
            (logits.sigmoid() >= threshold)
            if enabled
            else torch.zeros_like(logits, dtype=torch.bool)
        ).float()
        residual = (detectors.float() + correction @ self.h.T).remainder(2)
        prediction = (correction @ self.logical.T).remainder(2).to(torch.int8)
        return residual.to(torch.int8), prediction

    @torch.no_grad()
    def decode(
        self,
        detectors: torch.Tensor,
        backend: DetectorErrorModelDecoder,
        *,
        threshold: float = 0.9,
        enabled: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Combine the local correction's logical contribution with residual decode."""
        residual, prediction = self.correct(
            detectors, threshold=threshold, enabled=enabled, temperature=temperature
        )
        # The global DEM decoder consumes host detector indices explicitly.
        for i, row in enumerate(residual.cpu()):
            result = backend.decode(row.nonzero().flatten().tolist())
            for observable in result.observables:
                prediction[i, observable] ^= 1
        return prediction
