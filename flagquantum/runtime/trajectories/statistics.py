"""Online tensor statistics for trajectory observables."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch

from .result import TrajectoryStatistics


class TensorWelford:
    """Numerically stable online population moments for same-shaped tensors."""

    def __init__(self) -> None:
        self._count = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return self._count

    def update(self, value: torch.Tensor) -> None:
        value = torch.as_tensor(value)
        if self._mean is None:
            mean = value.clone()
            m2 = torch.zeros_like(value)
            self._count = 1
            self._mean = mean
            self._m2 = m2
            return
        if value.shape != self._mean.shape:
            raise ValueError(
                f"trajectory observable shape changed from {self._mean.shape} "
                f"to {value.shape}"
            )
        count = self._count + 1
        delta = value - self._mean
        mean = self._mean + delta / count
        assert self._m2 is not None
        m2 = self._m2 + delta * (value - mean)
        self._count = count
        self._mean = mean
        self._m2 = m2

    def merge(self, other: "TensorWelford") -> None:
        """Merge another accumulator without replaying its observations."""

        if other._count == 0:
            return
        if self._count == 0:
            self._count = other._count
            self._mean = other._mean
            self._m2 = other._m2
            return
        assert self._mean is not None and self._m2 is not None
        assert other._mean is not None and other._m2 is not None
        if self._mean.shape != other._mean.shape:
            raise ValueError(
                f"trajectory statistic shape mismatch: {self._mean.shape} "
                f"!= {other._mean.shape}"
            )
        other_mean = other._mean.to(device=self._mean.device, dtype=self._mean.dtype)
        other_m2 = other._m2.to(device=self._m2.device, dtype=self._m2.dtype)
        total = self._count + other._count
        delta = other_mean - self._mean
        self._mean = self._mean + delta * (other._count / total)
        self._m2 = (
            self._m2 + other_m2 + delta * delta * (self._count * other._count / total)
        )
        self._count = total

    def state_dict(self) -> dict[str, Any]:
        """Return a tensor-only state suitable for safe checkpointing."""

        return {
            "count": self._count,
            "mean": self._mean,
            "m2": self._m2,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TensorWelford":
        """Restore an accumulator with validation."""

        raw_count = state.get("count", 0)
        if not isinstance(raw_count, Integral) or isinstance(raw_count, bool):
            raise ValueError("trajectory statistic count must be an integer")
        count = int(raw_count)
        mean = state.get("mean")
        m2 = state.get("m2")
        if count < 0:
            raise ValueError("trajectory statistic count must be non-negative")
        if count == 0:
            if mean is not None or m2 is not None:
                raise ValueError("empty trajectory statistics cannot contain moments")
            return cls()
        if not isinstance(mean, torch.Tensor) or not isinstance(m2, torch.Tensor):
            raise ValueError("non-empty trajectory statistics require tensor moments")
        if mean.shape != m2.shape:
            raise ValueError("trajectory statistic moments must have the same shape")
        if mean.device != m2.device:
            raise ValueError("trajectory statistic moments must be on the same device")
        accumulator = cls()
        accumulator._count = count
        accumulator._mean = mean
        accumulator._m2 = m2
        return accumulator

    @classmethod
    def from_statistics(cls, statistics: TrajectoryStatistics) -> "TensorWelford":
        """Reconstruct mergeable moments from finalized population statistics."""

        if statistics.count <= 0:
            raise ValueError("trajectory statistics count must be positive")
        return cls.from_state_dict(
            {
                "count": statistics.count,
                "mean": statistics.mean,
                "m2": statistics.variance * statistics.count,
            }
        )

    def finalize(self) -> TrajectoryStatistics:
        if self._mean is None or self._m2 is None:
            raise ValueError("cannot finalize empty trajectory statistics")
        variance = self._m2 / self._count
        standard_error = torch.sqrt(
            torch.clamp(torch.real(variance), min=0) / self._count
        )
        return TrajectoryStatistics(
            count=self._count,
            mean=self._mean,
            variance=variance,
            standard_error=standard_error,
        )


__all__ = ("TensorWelford",)
