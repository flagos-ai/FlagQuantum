"""Asynchronous dtype-homogeneous gradient reduction buckets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class _PendingReduction:
    indices: tuple[int, ...]
    packed: torch.Tensor
    work: Any


class AsyncGradientReducer:
    """Launch ready gradient buckets while adjoint computation continues."""

    def __init__(
        self,
        gradients: list[torch.Tensor],
        *,
        process_group: Any | None,
        evidence: Any,
        max_parameters: int | None = None,
        max_bytes: int | None = None,
        async_op: bool = True,
    ) -> None:
        self.gradients = gradients
        self.process_group = process_group
        self.evidence = evidence
        self.max_parameters = int(
            max_parameters
            if max_parameters is not None
            else os.getenv("FQ_STATEVECTOR_GRADIENT_BUCKET_PARAMETERS", "32")
        )
        self.max_bytes = int(
            max_bytes
            if max_bytes is not None
            else os.getenv("FQ_STATEVECTOR_GRADIENT_BUCKET_BYTES", str(1 << 20))
        )
        self.async_op = bool(async_op)
        if self.max_parameters <= 0 or self.max_bytes <= 0:
            raise ValueError("gradient bucket limits must be positive")
        self._ready: dict[tuple[torch.device, torch.dtype], list[int]] = {}
        self._ready_bytes: dict[tuple[torch.device, torch.dtype], int] = {}
        self._pending: list[_PendingReduction] = []

    def mark_ready(self, index: int, *, overlap_opportunity: bool = True) -> None:
        gradient = self.gradients[int(index)]
        key = (gradient.device, gradient.dtype)
        indices = self._ready.setdefault(key, [])
        indices.append(int(index))
        byte_count = gradient.numel() * gradient.element_size()
        self._ready_bytes[key] = self._ready_bytes.get(key, 0) + byte_count
        if (
            len(indices) >= self.max_parameters
            or self._ready_bytes[key] >= self.max_bytes
        ):
            self._launch(key, before_adjoint_complete=overlap_opportunity)

    def _launch(
        self,
        key: tuple[torch.device, torch.dtype],
        *,
        before_adjoint_complete: bool,
    ) -> None:
        indices = tuple(self._ready.pop(key, ()))
        self._ready_bytes.pop(key, None)
        if not indices:
            return
        packed = torch.cat([self.gradients[index].reshape(-1) for index in indices])
        work = dist.all_reduce(
            packed,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=self.async_op,
        )
        byte_count = packed.numel() * packed.element_size()
        self.evidence.communication_count += 1
        self.evidence.communication_bytes += byte_count
        self.evidence.gradient_collective_count += 1
        self.evidence.gradient_collective_bytes += byte_count
        if self.async_op:
            self.evidence.async_gradient_collective_count += 1
        if self.async_op and before_adjoint_complete:
            self.evidence.overlapped_gradient_collective_count += 1
        if self.async_op:
            self._pending.append(_PendingReduction(indices, packed, work))
        else:
            self._scatter(indices, packed)

    def _scatter(self, indices: tuple[int, ...], packed: torch.Tensor) -> None:
        offset = 0
        for index in indices:
            count = self.gradients[index].numel()
            self.gradients[index].copy_(
                packed[offset : offset + count].reshape_as(self.gradients[index])
            )
            offset += count

    def finish(self) -> None:
        for key in tuple(self._ready):
            self._launch(key, before_adjoint_complete=False)
        for pending in self._pending:
            pending.work.wait()
            self._scatter(pending.indices, pending.packed)


__all__ = ("AsyncGradientReducer",)
