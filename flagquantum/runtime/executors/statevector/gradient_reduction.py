"""Dtype-homogeneous gradient reduction, replicated or owner-sharded.

Two reduction schemes are available, and the difference is a property of the
collective that ran rather than a label attached afterwards.

``all_reduce`` gives every rank the whole gradient vector, so a rank that steps
only its own parameters still holds every other rank's gradients. That is
replicated gradient ownership, and it is what the asynchronous bucketed
reducer below reports.

``reduce_scatter_tensor`` computes the same sum and leaves each rank holding
only its own parameters' block. Owner-sharded gradients are therefore what the
collective did, which is what allows a training result to report
``gradient_ownership_semantics="sharded_across_ranks"`` without asserting
anything the communication did not do.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

#: Gradient ownership left every rank holding the entire reduced vector.
REPLICATED_ALL_REDUCE = "replicated_all_reduce"
#: Gradient ownership left each rank holding only the parameters it owns.
OWNER_SHARDED_REDUCE_SCATTER = "owner_sharded_reduce_scatter"


@dataclass
class _PendingReduction:
    indices: tuple[int, ...]
    packed: torch.Tensor
    work: Any


class AsyncGradientReducer:
    """Reduce accumulated gradients, replicated or onto their owner ranks.

    With ``owners`` supplied, every gradient is reduced exactly once into the
    block belonging to the rank that owns it, including parameters whose owner
    is not this rank. Those local buffers are zeroed, because the owner's step
    writes the true value back and a stale partial sum must not be mistaken for
    a gradient.
    """

    def __init__(
        self,
        gradients: list[torch.Tensor],
        *,
        process_group: Any | None,
        evidence: Any,
        owners: Sequence[int] | None = None,
        max_parameters: int | None = None,
        max_bytes: int | None = None,
        async_op: bool = True,
    ) -> None:
        self.gradients = gradients
        self.process_group = process_group
        self.evidence = evidence
        self.owners = None if owners is None else tuple(int(item) for item in owners)
        if self.owners is not None and len(self.owners) != len(gradients):
            raise ValueError(
                "owner-sharded reduction needs one owner per gradient, "
                f"received {len(self.owners)} owners for {len(gradients)} gradients"
            )
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
        if self.owners is not None:
            # One reduce-scatter covers every rank's block at once, so there is
            # no bucket to launch early and no overlap for an async handle to buy.
            self.async_op = False
        if self.max_parameters <= 0 or self.max_bytes <= 0:
            raise ValueError("gradient bucket limits must be positive")
        self._ready: dict[tuple[torch.device, torch.dtype], list[int]] = {}
        self._ready_bytes: dict[tuple[torch.device, torch.dtype], int] = {}
        self._pending: list[_PendingReduction] = []
        self._marked: list[int] = []

    def mark_ready(self, index: int, *, overlap_opportunity: bool = True) -> None:
        gradient = self.gradients[int(index)]
        if self.owners is not None:
            # One reduce-scatter covers every rank's block at once, so there is
            # nothing to launch early and nothing for readiness order to buy.
            self._marked.append(int(index))
            return
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

    def _marked_groups(self) -> list[tuple[torch.dtype, torch.device, list[int]]]:
        """Marked gradients grouped by dtype and device, in stable index order.

        A single ``reduce_scatter_tensor`` call carries one dtype on one device,
        so the groups are what the collective boundary has to be.
        """

        groups: dict[tuple[torch.dtype, torch.device], list[int]] = {}
        for index in sorted(set(self._marked)):
            gradient = self.gradients[index]
            groups.setdefault((gradient.dtype, gradient.device), []).append(index)
        return [
            (dtype, device, indices)
            for (dtype, device), indices in sorted(
                groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
            )
        ]

    def _reduce_scatter(self) -> None:
        """Leave every rank holding only its own parameters' reduced gradient."""

        owners = self.owners
        assert owners is not None
        world_size = dist.get_world_size(self.process_group)
        rank = dist.get_rank(self.process_group)
        for dtype, device, indices in self._marked_groups():
            # Every rank's block has to occupy the same number of elements, so
            # the widest owner block sets the segment and the others pad. A
            # padded lane carries zeros, which the sum leaves unchanged.
            per_rank_amplitudes = [
                sum(
                    self.gradients[index].numel()
                    for index in indices
                    if owners[index] == owner
                )
                for owner in range(world_size)
            ]
            segment = max(per_rank_amplitudes, default=0)
            if segment == 0:
                continue
            packed = torch.zeros(world_size * segment, dtype=dtype, device=device)
            cursor = [owner * segment for owner in range(world_size)]
            owned_here: list[int] = []
            for index in indices:
                flat = self.gradients[index].reshape(-1)
                owner = owners[index]
                packed[cursor[owner] : cursor[owner] + flat.numel()] = flat
                cursor[owner] += flat.numel()
                if owner == rank:
                    owned_here.append(index)
            received = torch.empty(segment, dtype=dtype, device=device)
            dist.reduce_scatter_tensor(
                received,
                packed,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
            byte_count = packed.numel() * packed.element_size()
            self.evidence.communication_count += 1
            self.evidence.communication_bytes += byte_count
            self.evidence.gradient_collective_count += 1
            self.evidence.gradient_collective_bytes += byte_count
            # The received segment holds this rank's own parameters, packed in
            # the same order they were written, so the cursor walks it forward.
            offset = 0
            for index in owned_here:
                count = self.gradients[index].numel()
                self.gradients[index].copy_(
                    received[offset : offset + count].reshape_as(self.gradients[index])
                )
                offset += count
            for index in indices:
                if owners[index] != rank:
                    self.gradients[index].zero_()
        self.evidence.gradient_reduction_scheme = OWNER_SHARDED_REDUCE_SCATTER

    def finish(self) -> None:
        if self.owners is not None:
            self._reduce_scatter()
            return
        for key in tuple(self._ready):
            self._launch(key, before_adjoint_complete=False)
        for pending in self._pending:
            pending.work.wait()
            self._scatter(pending.indices, pending.packed)
        self.evidence.gradient_reduction_scheme = REPLICATED_ALL_REDUCE


__all__ = (
    "OWNER_SHARDED_REDUCE_SCATTER",
    "REPLICATED_ALL_REDUCE",
    "AsyncGradientReducer",
)
