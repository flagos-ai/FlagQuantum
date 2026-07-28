"""Bounded preparation of compiled rank-local MPS gate layers."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ....core.ir import Instruction
from ....simulation.mps import _split_pair_matrix_bucket
from .communication import _instruction_matrix_for_mps
from .errors import MPSForwardLifetimeError
from .factorization import (
    FactorizationWorkspacePolicy,
    MemoryProvider,
    factorization_workspace_pool,
    plan_rxx_factorization_microbatch,
)
from .site_kernels import (
    apply_rxx_contraction_bucket,
    apply_ry_bucket,
    site_kernel_bucket_capacity,
)
from .state import RankOwnedMPSState


def require_layer_cache_drained(
    precomputed: dict[int, tuple[torch.Tensor, ...]],
    *,
    layer_sequence: int,
    layer_start: int,
    layer_end: int,
) -> None:
    """Fail before the next layer if a prepared tensor missed its last use."""
    if not precomputed:
        return
    retained = tuple(sorted(int(index) for index in precomputed))
    raise MPSForwardLifetimeError(
        "compiled MPS layer cache was not drained: "
        f"layer={layer_sequence}, instructions={layer_start}:{layer_end}, "
        f"retained={retained}"
    )


def device_memory_metadata(device: torch.device) -> dict[str, int | None]:
    """Return JSON-safe allocator observations for one execution device."""
    if device.type != "cuda":
        return {
            "allocated_memory_bytes": None,
            "reserved_memory_bytes": None,
            "peak_allocated_memory_bytes": None,
        }
    return {
        "allocated_memory_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_memory_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def prepare_compiled_mps_layer(
    instructions: Sequence[Instruction],
    *,
    layer_start: int,
    state: RankOwnedMPSState,
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
    factorization_workspace_policy: FactorizationWorkspacePolicy,
    factorization_memory_provider: MemoryProvider,
) -> tuple[
    tuple[tuple[int, Instruction, tuple[int, ...]], ...],
    dict[int, tuple[torch.Tensor, ...]],
    tuple[dict[str, Any], ...],
]:
    """Prepare one disjoint rotation layer in a bounded tensor scope."""
    first = instructions[layer_start]
    layer: list[tuple[int, Instruction, tuple[int, ...]]] = []
    used: set[int] = set()
    for index in range(layer_start, len(instructions)):
        candidate = instructions[index]
        wires = tuple(int(wire) for wire in candidate.wires)
        if candidate.name != first.name or used.intersection(wires):
            break
        used.update(wires)
        layer.append((index, candidate, wires))

    precomputed: dict[int, tuple[torch.Tensor, ...]] = {}
    factorization_records: list[dict[str, Any]] = []
    if first.name == "ry":
        buckets: dict[
            tuple[int, ...], list[tuple[int, Instruction, tuple[int, ...]]]
        ] = {}
        for item in layer:
            _, _, wires = item
            if state.owner(wires[0]) == state.rank:
                key = tuple(state.local_tensors[wires[0]].shape)
                buckets.setdefault(key, []).append(item)
        for bucket in buckets.values():
            _, sample, sample_wires = bucket[0]
            sample_matrix = _instruction_matrix_for_mps(
                sample, bsz=bsz, device=device, dtype=dtype
            )
            if sample_matrix.ndim == 2:
                sample_matrix = sample_matrix.expand(bsz, -1, -1)
            capacity = site_kernel_bucket_capacity(
                state.local_tensors[sample_wires[0]], sample_matrix
            )
            for start in range(0, len(bucket), capacity):
                chunk = bucket[start : start + capacity]
                packed = torch.stack(
                    [state.local_tensors[wires[0]] for _, _, wires in chunk]
                )
                packed_matrices = []
                for _, candidate, _ in chunk:
                    matrix = _instruction_matrix_for_mps(
                        candidate, bsz=bsz, device=device, dtype=dtype
                    )
                    packed_matrices.append(
                        matrix.expand(bsz, -1, -1) if matrix.ndim == 2 else matrix
                    )
                updated = apply_ry_bucket(
                    packed, torch.stack(packed_matrices), compiled=True
                )
                for position, (index, _, wires) in enumerate(chunk):
                    value = updated[position]
                    precomputed[index] = (value,)
                    state.local_tensors[wires[0]] = value
    elif first.name in {"rxx", "ryy", "rzz"}:
        buckets: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            list[tuple[int, Instruction, tuple[int, ...]]],
        ] = {}
        for item in layer:
            _, _, wires = item
            left = min(wires)
            if state.owner(left) == state.rank and state.owner(left + 1) == state.rank:
                key = (
                    tuple(state.local_tensors[left].shape),
                    tuple(state.local_tensors[left + 1].shape),
                )
                buckets.setdefault(key, []).append(item)
        for bucket in buckets.values():
            _, sample, sample_wires = bucket[0]
            sample_left = min(sample_wires)
            sample_matrix = _instruction_matrix_for_mps(
                sample, bsz=bsz, device=device, dtype=dtype
            )
            if sample_matrix.ndim == 2:
                sample_matrix = sample_matrix.expand(bsz, -1, -1)
            capacity = site_kernel_bucket_capacity(
                state.local_tensors[sample_left],
                state.local_tensors[sample_left + 1],
                sample_matrix,
            )
            start = 0
            while start < len(bucket):
                requested = min(capacity, len(bucket) - start)
                decision = plan_rxx_factorization_microbatch(
                    state.local_tensors[sample_left],
                    state.local_tensors[sample_left + 1],
                    sample_matrix,
                    requested_chunk_size=requested,
                    policy=factorization_workspace_policy,
                    memory_provider=factorization_memory_provider,
                )
                chunk = bucket[start : start + decision.selected_chunk_size]
                factorization_records.append(
                    {
                        **decision.as_dict(),
                        "layer_start": layer_start,
                        "bucket_item_start": start,
                    }
                )
                left_values = [
                    state.local_tensors[min(wires)] for _, _, wires in chunk
                ]
                right_values = [
                    state.local_tensors[min(wires) + 1] for _, _, wires in chunk
                ]
                pool = factorization_workspace_pool()
                left_entry = pool.acquire(
                    role="rxx_left",
                    shape=(len(left_values), *left_values[0].shape),
                    dtype=left_values[0].dtype,
                    device=left_values[0].device,
                )
                right_entry = pool.acquire(
                    role="rxx_right",
                    shape=(len(right_values), *right_values[0].shape),
                    dtype=right_values[0].dtype,
                    device=right_values[0].device,
                )
                lefts = left_entry.tensor
                rights = right_entry.tensor
                try:
                    for position, value in enumerate(left_values):
                        lefts[position].copy_(value)
                    for position, value in enumerate(right_values):
                        rights[position].copy_(value)
                    packed_matrices = []
                    for _, candidate, _ in chunk:
                        matrix = _instruction_matrix_for_mps(
                            candidate, bsz=bsz, device=device, dtype=dtype
                        )
                        packed_matrices.append(
                            matrix.expand(bsz, -1, -1)
                            if matrix.ndim == 2
                            else matrix
                        )
                    pairs = apply_rxx_contraction_bucket(
                        lefts,
                        rights,
                        torch.stack(packed_matrices),
                        compiled=True,
                    )
                    split_outputs = _split_pair_matrix_bucket(
                        pairs,
                        left_dim=lefts.shape[2],
                        right_dim=rights.shape[-1],
                        config=state.config,
                        svd_driver=factorization_workspace_policy.svd_driver,
                    )
                finally:
                    pool.release(left_entry)
                    pool.release(right_entry)
                factorization_records[-1]["staging_workspace_pool"] = pool.stats()
                for position, (index, _, wires) in enumerate(chunk):
                    left = min(wires)
                    updated_left, updated_right, info = split_outputs[position]
                    precomputed[index] = (updated_left, updated_right, info)
                    state.local_tensors[left] = updated_left
                    state.local_tensors[left + 1] = updated_right
                start += decision.selected_chunk_size
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unsupported compiled MPS layer {first.name!r}")
    return tuple(layer), precomputed, tuple(factorization_records)


__all__ = (
    "device_memory_metadata",
    "prepare_compiled_mps_layer",
    "require_layer_cache_drained",
)
