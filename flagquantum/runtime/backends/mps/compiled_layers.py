"""Bounded preparation of compiled rank-local MPS gate layers."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ....core.ir import Instruction
from ....simulation.mps_compiled_layers import (
    apply_compiled_mps_one_site_bucket,
    apply_compiled_mps_two_site_bucket,
)
from ....simulation.mps_site_kernels import site_kernel_bucket_capacity
from .communication import _instruction_matrix_for_mps
from .errors import MPSForwardLifetimeError
from .factorization import (
    FactorizationWorkspacePolicy,
    MemoryProvider,
    factorization_workspace_pool,
    plan_rxx_factorization_microbatch,
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
                updated = apply_compiled_mps_one_site_bucket(
                    tuple(candidate for _, candidate, _ in chunk),
                    tuple(state.local_tensors[wires[0]] for _, _, wires in chunk),
                    bsz=bsz,
                    device=device,
                    dtype=dtype,
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
                isolate_factorizations = (
                    state.config.max_bond is not None
                    and int(state.config.max_bond) >= 128
                )
                split_outputs = apply_compiled_mps_two_site_bucket(
                    tuple(candidate for _, candidate, _ in chunk),
                    tuple(state.local_tensors[min(wires)] for _, _, wires in chunk),
                    tuple(state.local_tensors[min(wires) + 1] for _, _, wires in chunk),
                    state.config,
                    bsz=bsz,
                    device=device,
                    dtype=dtype,
                    isolate_factorizations=isolate_factorizations,
                    svd_driver=factorization_workspace_policy.svd_driver,
                )
                factorization_records[-1]["batched_contraction_enabled"] = True
                factorization_records[-1][
                    "batched_factorization_enabled"
                ] = not isolate_factorizations
                if isolate_factorizations:
                    factorization_records[-1][
                        "reason"
                    ] = "high_bond_batched_factorization_quarantine"
                factorization_records[-1]["staging_workspace_pool"] = {
                    "enabled": False,
                    "reason": "staging_pool_correctness_quarantine",
                    **factorization_workspace_pool().stats(),
                }
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
