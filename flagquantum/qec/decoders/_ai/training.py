"""Private device training and exact optimizer/batch-stream checkpoint recovery."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from flagquantum.qec.decoders._ai.predecoder import Predecoder, _positive


class Trainer:
    """Adam training with exact device batch-stream and optimizer resume."""

    def __init__(
        self, model: Predecoder, *, seed: int = 0, learning_rate: float = 0.001
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.generator = torch.Generator(device=model.prior.device).manual_seed(seed)
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
