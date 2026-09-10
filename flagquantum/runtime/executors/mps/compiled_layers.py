"""Bounded preparation of compiled rank-local MPS gate layers."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ....compute import get_platform_runtime
from ....core.ir import Instruction
from ....simulation.mps.compiled_layers import (
    apply_compiled_mps_one_site_bucket,
    apply_compiled_mps_two_site_bucket,
)
from ....simulation.mps.rank_local import (
    instruction_matrix_for_mps as _instruction_matrix_for_mps,
)
from ....simulation.mps.site_kernels import site_kernel_bucket_capacity
from .errors import MPSForwardLifetimeError
from .factorization import (
    FactorizationWorkspacePolicy,
    MemoryProvider,
    factorization_workspace_pool,
    plan_rxx_factorization_microbatch,
)
from .state import RankOwnedMPSState

_PreparedMPSOutput = (
    tuple[torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]
)
_MPSLayerItem = tuple[int, Instruction, tuple[int, ...]]


def require_layer_cache_drained(
    precomputed: dict[int, _PreparedMPSOutput],
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
    memory = get_platform_runtime(device.type).memory_snapshot(device)
    return {
        "allocated_memory_bytes": memory.allocated_bytes,
        "reserved_memory_bytes": memory.reserved_bytes,
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _collect_compiled_layer(
    instructions: Sequence[Instruction], layer_start: int
) -> tuple[_MPSLayerItem, ...]:
    first = instructions[layer_start]
    layer: list[_MPSLayerItem] = []
    used: set[int] = set()
    for index in range(layer_start, len(instructions)):
        candidate = instructions[index]
        wires = tuple(int(wire) for wire in candidate.wires)
        if candidate.name != first.name or used.intersection(wires):
            break
        used.update(wires)
        layer.append((index, candidate, wires))
    return tuple(layer)


def _prepare_one_site_layer(
    layer: Sequence[_MPSLayerItem],
    *,
    state: RankOwnedMPSState,
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, _PreparedMPSOutput]:
    precomputed: dict[int, _PreparedMPSOutput] = {}
    buckets: dict[tuple[int, ...], list[_MPSLayerItem]] = {}
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
    return precomputed


def _prepare_two_site_layer(
    layer: Sequence[_MPSLayerItem],
    *,
    layer_start: int,
    state: RankOwnedMPSState,
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
    factorization_workspace_policy: FactorizationWorkspacePolicy,
    factorization_memory_provider: MemoryProvider,
) -> tuple[dict[int, _PreparedMPSOutput], list[dict[str, Any]]]:
    precomputed: dict[int, _PreparedMPSOutput] = {}
    factorization_records: list[dict[str, Any]] = []
    buckets: dict[tuple[tuple[int, ...], tuple[int, ...]], list[_MPSLayerItem]] = {}
    for item in layer:
        _, _, wires = item
        left = min(wires)
        if state.owner(left) == state.rank and state.owner(left + 1) == state.rank:
            shape_pair = (
                tuple(state.local_tensors[left].shape),
                tuple(state.local_tensors[left + 1].shape),
            )
            buckets.setdefault(shape_pair, []).append(item)
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
                state.config.max_bond is not None and int(state.config.max_bond) >= 128
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
            record = factorization_records[-1]
            record["batched_contraction_enabled"] = True
            record["batched_factorization_enabled"] = not isolate_factorizations
            if isolate_factorizations:
                record["reason"] = "high_bond_batched_factorization_quarantine"
            record["staging_workspace_pool"] = {
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
    return precomputed, factorization_records


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
    tuple[_MPSLayerItem, ...],
    dict[int, _PreparedMPSOutput],
    tuple[dict[str, Any], ...],
]:
    """Prepare one disjoint rotation layer in a bounded tensor scope."""
    layer = _collect_compiled_layer(instructions, layer_start)
    if layer[0][1].name == "ry":
        precomputed = _prepare_one_site_layer(
            layer, state=state, bsz=bsz, device=device, dtype=dtype
        )
        factorization_records: list[dict[str, Any]] = []
    elif layer[0][1].name in {"rxx", "ryy", "rzz"}:
        precomputed, factorization_records = _prepare_two_site_layer(
            layer,
            layer_start=layer_start,
            state=state,
            bsz=bsz,
            device=device,
            dtype=dtype,
            factorization_workspace_policy=factorization_workspace_policy,
            factorization_memory_provider=factorization_memory_provider,
        )
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unsupported compiled MPS layer {layer[0][1].name!r}")
    return layer, precomputed, tuple(factorization_records)


__all__ = (
    "device_memory_metadata",
    "prepare_compiled_mps_layer",
    "require_layer_cache_drained",
)
