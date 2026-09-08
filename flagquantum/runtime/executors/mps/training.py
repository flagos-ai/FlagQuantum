"""Owner-sharded MPS training entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .training_records import ShardedMPSTrainingResult


def train_distributed_mps(*args: Any, **kwargs: Any) -> ShardedMPSTrainingResult:
    """Run owner-sharded training through the canonical execution engine."""

    from .training_engine import train_distributed_mps as execute

    return execute(*args, **kwargs)


__all__ = ("train_distributed_mps",)
