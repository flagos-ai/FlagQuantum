"""Deterministic validation and execution planning for MPS reverse mode."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch

from .records import (
    MPSReverseContractError,
    MPSReverseTape,
    MPSReverseTapeRecord,
)

_REVERSE_SEGMENT_CACHE_MAX_ENTRIES = 128
_REVERSE_SEGMENT_CACHE: OrderedDict[tuple[str, bool], tuple[tuple[int, ...], ...]] = (
    OrderedDict()
)
_REVERSE_SEGMENT_CACHE_HITS = 0
_REVERSE_SEGMENT_CACHE_MISSES = 0
_GRADIENT_BUCKET_CACHE_MAX_ENTRIES = 128
_GRADIENT_BUCKET_CACHE: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()


def validate_mps_svd_gaps(
    tape: MPSReverseTape, tolerance: float, *, allow_degenerate: bool = False
) -> None:
    """Reject unstable truncated-SVD pullbacks unless explicitly allowed."""
    unstable = tuple(
        record
        for record in tape.records
        if record.singular_value_gap is not None
        and record.singular_value_gap <= tolerance
        and record.discarded_weight > tolerance * tolerance
    )
    if unstable and not allow_degenerate:
        raise MPSReverseContractError(
            "truncated SVD pullback is degenerate at operation(s): "
            + ",".join(record.operation_id for record in unstable)
        )


def plan_mps_reverse_segments(
    tape: MPSReverseTape, *, fuse_owner_local: bool
) -> tuple[tuple[MPSReverseTapeRecord, ...], ...]:
    """Partition reverse order into safe owner-local one-site VJP segments."""
    ordered = tuple(reversed(tape.records))
    if not fuse_owner_local:
        return tuple((record,) for record in ordered)
    segments: list[tuple[MPSReverseTapeRecord, ...]] = []
    pending: list[MPSReverseTapeRecord] = []
    wires: set[int] = set()
    signature = None

    def flush() -> None:
        nonlocal pending, wires, signature
        if pending:
            segments.append(tuple(pending))
        pending, wires, signature = [], set(), None

    for record in ordered:
        candidate = (
            record.kind == "one_site"
            and record.communication_peer is None
            and len(record.input_shapes) == 1
        )
        current = (record.compute_owner, record.input_shapes, record.output_shapes)
        wire = record.wires[0]
        if not candidate or (pending and (current != signature or wire in wires)):
            flush()
        if candidate:
            pending.append(record)
            wires.add(wire)
            signature = current
        else:
            segments.append((record,))
    flush()
    return tuple(segments)


def cached_mps_reverse_segments(
    tape: MPSReverseTape, *, fuse_owner_local: bool
) -> tuple[tuple[tuple[MPSReverseTapeRecord, ...], ...], bool]:
    """Resolve a bounded identity-keyed plan without retaining old tensors.

    Only record indices are cached.  Every call maps those indices onto the
    current tape, so the cache cannot retain payloads, parameters, or prior-step
    tensor storage.
    """
    global _REVERSE_SEGMENT_CACHE_HITS, _REVERSE_SEGMENT_CACHE_MISSES
    key = (tape.identity, bool(fuse_owner_local))
    indices = _REVERSE_SEGMENT_CACHE.get(key)
    hit = indices is not None
    if hit:
        _REVERSE_SEGMENT_CACHE_HITS += 1
        _REVERSE_SEGMENT_CACHE.move_to_end(key)
    else:
        _REVERSE_SEGMENT_CACHE_MISSES += 1
        planned = plan_mps_reverse_segments(tape, fuse_owner_local=fuse_owner_local)
        by_identity = {id(record): index for index, record in enumerate(tape.records)}
        indices = tuple(
            tuple(by_identity[id(record)] for record in segment) for segment in planned
        )
        _REVERSE_SEGMENT_CACHE[key] = indices
        while len(_REVERSE_SEGMENT_CACHE) > _REVERSE_SEGMENT_CACHE_MAX_ENTRIES:
            _REVERSE_SEGMENT_CACHE.popitem(last=False)
    return (
        tuple(tuple(tape.records[index] for index in segment) for segment in indices),
        hit,
    )


def clear_mps_reverse_segment_cache() -> None:
    global _REVERSE_SEGMENT_CACHE_HITS, _REVERSE_SEGMENT_CACHE_MISSES
    _REVERSE_SEGMENT_CACHE.clear()
    _REVERSE_SEGMENT_CACHE_HITS = 0
    _REVERSE_SEGMENT_CACHE_MISSES = 0
    _GRADIENT_BUCKET_CACHE.clear()


def mps_reverse_segment_cache_stats() -> dict[str, int]:
    return {
        "entries": len(_REVERSE_SEGMENT_CACHE),
        "max_entries": _REVERSE_SEGMENT_CACHE_MAX_ENTRIES,
        "hits": _REVERSE_SEGMENT_CACHE_HITS,
        "misses": _REVERSE_SEGMENT_CACHE_MISSES,
    }


def plan_mps_gradient_buckets(
    parameters: Sequence[torch.Tensor],
    owners: Sequence[int] | None,
    *,
    max_bucket_bytes: int,
) -> tuple[tuple[torch.dtype, int | None, tuple[tuple[int, int, int], ...]], ...]:
    """Create deterministic bounded dtype/owner/writeback gradient buckets."""
    if max_bucket_bytes <= 0:
        raise ValueError("gradient_bucket_bytes must be positive")
    if owners is not None and len(owners) != len(parameters):
        raise ValueError("gradient_owner_ranks must match trainable parameters")
    grouped: dict[tuple[torch.dtype, int | None], list[tuple[int, int, int]]] = {}
    for index, parameter in enumerate(parameters):
        owner = None if owners is None else int(owners[index])
        elements_per_bucket = max(1, max_bucket_bytes // parameter.element_size())
        for start in range(0, parameter.numel(), elements_per_bucket):
            grouped.setdefault((parameter.dtype, owner), []).append(
                (index, start, min(parameter.numel(), start + elements_per_bucket))
            )
    buckets = []
    for (dtype, owner), pieces in grouped.items():
        current, current_bytes = [], 0
        element_size = torch.empty((), dtype=dtype).element_size()
        for piece in pieces:
            piece_bytes = (piece[2] - piece[1]) * element_size
            if current and current_bytes + piece_bytes > max_bucket_bytes:
                buckets.append((dtype, owner, tuple(current)))
                current, current_bytes = [], 0
            current.append(piece)
            current_bytes += piece_bytes
        if current:
            buckets.append((dtype, owner, tuple(current)))
    return tuple(buckets)


def cached_mps_gradient_buckets(
    parameters: Sequence[torch.Tensor],
    owners: Sequence[int] | None,
    *,
    max_bucket_bytes: int,
) -> tuple[
    tuple[tuple[torch.dtype, int | None, tuple[tuple[int, int, int], ...]], ...],
    bool,
]:
    """Cache a tensor-free layout by dtype, size, owner, and byte bound."""
    owner_key = None if owners is None else tuple(int(owner) for owner in owners)
    key = (
        tuple((parameter.dtype, parameter.numel()) for parameter in parameters),
        owner_key,
        int(max_bucket_bytes),
    )
    cached = _GRADIENT_BUCKET_CACHE.get(key)
    if cached is not None:
        _GRADIENT_BUCKET_CACHE.move_to_end(key)
        return cached, True
    planned = plan_mps_gradient_buckets(
        parameters, owners, max_bucket_bytes=max_bucket_bytes
    )
    _GRADIENT_BUCKET_CACHE[key] = planned
    while len(_GRADIENT_BUCKET_CACHE) > _GRADIENT_BUCKET_CACHE_MAX_ENTRIES:
        _GRADIENT_BUCKET_CACHE.popitem(last=False)
    return planned, False


def plan_mps_canonicalization_bonds(
    n_wires: int, dirty_bonds: Sequence[int], policy: str
) -> tuple[int, ...]:
    """Plan the minimal canonicalization interval for the declared policy."""
    if policy not in {"none", "dirty", "full"}:
        raise ValueError("canonicalization_policy must be none, dirty or full")
    if policy == "none":
        return ()
    if policy == "full":
        return tuple(range(max(0, n_wires - 1)))
    dirty = tuple(sorted(set(int(bond) for bond in dirty_bonds)))
    if any(bond < 0 or bond >= n_wires - 1 for bond in dirty):
        raise ValueError("dirty canonicalization bond is outside the MPS")
    return () if not dirty else tuple(range(dirty[0], n_wires - 1))


def discover_trainable_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    """Find unique trainable leaves reachable from a nested tensor value."""
    found: dict[int, torch.Tensor] = {}
    visited: set[int] = set()

    def visit_function(function: Any) -> None:
        if function is None or id(function) in visited:
            return
        visited.add(id(function))
        variable = getattr(function, "variable", None)
        if isinstance(variable, torch.Tensor) and variable.requires_grad:
            found.setdefault(id(variable), variable)
        for next_function, _ in getattr(function, "next_functions", ()):
            visit_function(next_function)

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)
            return
        if not isinstance(item, torch.Tensor) or not item.requires_grad:
            return
        if item.is_leaf:
            found.setdefault(id(item), item)
            return
        visit_function(item.grad_fn)

    visit(value)
    return tuple(found.values())


def build_mps_parameter_layout(
    ir: Any,
) -> tuple[tuple[torch.Tensor, ...], tuple[tuple[int, ...], ...]]:
    """Build stable global parameters and per-instruction parameter indices."""
    parameters: list[torch.Tensor] = []
    identities: dict[int, int] = {}
    instruction_indices = []
    for instruction in ir.instructions:
        leaves = discover_trainable_tensors(
            instruction.params
        ) + discover_trainable_tensors(instruction.matrix)
        active = []
        for leaf in leaves:
            index = identities.get(id(leaf))
            if index is None:
                index = len(parameters)
                identities[id(leaf)] = index
                parameters.append(leaf)
            if index not in active:
                active.append(index)
        instruction_indices.append(tuple(active))
    if not parameters:
        raise ValueError("sharded MPS reverse requires at least one trainable tensor")
    return tuple(parameters), tuple(instruction_indices)


__all__ = (
    "cached_mps_gradient_buckets",
    "cached_mps_reverse_segments",
    "clear_mps_reverse_segment_cache",
    "plan_mps_canonicalization_bonds",
    "build_mps_parameter_layout",
    "discover_trainable_tensors",
    "plan_mps_gradient_buckets",
    "plan_mps_reverse_segments",
    "mps_reverse_segment_cache_stats",
    "validate_mps_svd_gaps",
)
