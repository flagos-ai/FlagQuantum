"""State-owning P2P transport for distributed MPS execution."""

from __future__ import annotations

import datetime
import math
import os
import time
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from ....compute import get_platform_runtime


def _p2p_timeout() -> datetime.timedelta:
    seconds = float(os.environ.get("FLAGQUANTUM_MPS_P2P_TIMEOUT_SECONDS", "120"))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            "FLAGQUANTUM_MPS_P2P_TIMEOUT_SECONDS must be positive and finite"
        )
    return datetime.timedelta(seconds=seconds)


def _p2p_state() -> str:
    if not dist.is_initialized():
        return "process_group=uninitialized"
    return (
        f"process_group=initialized,backend={dist.get_backend()},"
        f"rank={dist.get_rank()},world_size={dist.get_world_size()}"
    )


def _wait_batched_p2p(operations: list[dist.P2POp], *, diagnostic: str) -> None:
    timeout = _p2p_timeout()
    started = time.perf_counter()
    _MPS_P2P_STATS["physical_message_count"] += len(operations)
    for request in dist.batch_isend_irecv(operations):
        try:
            completed = request.wait(timeout=timeout)
        except Exception as exc:
            raise RuntimeError(
                f"MPS P2P failed: {diagnostic}; {_p2p_state()}; "
                f"timeout_seconds={timeout.total_seconds():g}; cause={exc}"
            ) from exc
        if completed is False:
            raise RuntimeError(
                f"MPS P2P timeout: {diagnostic}; {_p2p_state()}; "
                f"timeout_seconds={timeout.total_seconds():g}"
            )
    _MPS_P2P_STATS["host_wait_seconds"] += time.perf_counter() - started


_MPS_P2P_STREAMS: dict[tuple[str, int | None], torch.cuda.Stream] = {}
_WARMED_MPS_NEIGHBORS: set[tuple[int, int, int, str]] = set()
_MPS_P2P_BUFFER_POOL: dict[tuple[str, int | None, torch.dtype, int], torch.Tensor] = {}
_MPS_STATIC_DESCRIPTOR_CACHE: dict[
    tuple[int, int, int], tuple[int, tuple[int, ...], torch.dtype]
] = {}
_MPS_P2P_STATS: dict[str, float | int] = {
    "physical_message_count": 0,
    "logical_tensor_count": 0,
    "logical_payload_bytes": 0,
    "host_wait_seconds": 0.0,
    "overlap_work_seconds": 0.0,
    "buffer_pool_hits": 0,
    "buffer_pool_misses": 0,
    "buffer_pool_bytes": 0,
    "descriptor_message_count": 0,
    "descriptor_cache_hits": 0,
    "descriptor_cache_misses": 0,
    "descriptor_cache_invalidations": 0,
    "static_payload_message_count": 0,
}


def reset_mps_p2p_stats() -> None:
    for key in tuple(_MPS_P2P_STATS):
        _MPS_P2P_STATS[key] = 0.0 if "seconds" in key else 0


def mps_p2p_stats() -> dict[str, float | int]:
    return dict(_MPS_P2P_STATS)


def clear_mps_static_descriptor_cache() -> None:
    _MPS_STATIC_DESCRIPTOR_CACHE.clear()


def mps_static_descriptor_cache_entries() -> int:
    return len(_MPS_STATIC_DESCRIPTOR_CACHE)


def _pooled_p2p_buffer(
    *, device: torch.device, dtype: torch.dtype, elements: int
) -> torch.Tensor:
    key = (device.type, device.index, dtype, int(elements))
    value = _MPS_P2P_BUFFER_POOL.get(key)
    if value is not None:
        _MPS_P2P_STATS["buffer_pool_hits"] += 1
        return value
    value = torch.empty(elements, dtype=dtype, device=device)
    _MPS_P2P_BUFFER_POOL[key] = value
    _MPS_P2P_STATS["buffer_pool_misses"] += 1
    _MPS_P2P_STATS["buffer_pool_bytes"] += value.numel() * value.element_size()
    return value


def _run_batched_p2p(
    operations: list[dist.P2POp], device: torch.device, *, diagnostic: str
) -> None:
    if device.type != "cuda":
        _wait_batched_p2p(operations, diagnostic=diagnostic)
        return
    key = (device.type, device.index)
    stream = _MPS_P2P_STREAMS.get(key)
    if stream is None:
        stream = get_platform_runtime(device.type).stream(device)
        _MPS_P2P_STREAMS[key] = stream
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        _wait_batched_p2p(operations, diagnostic=diagnostic)
    current.wait_stream(stream)


def _run_batched_p2p_with_overlap(
    operations: list[dist.P2POp],
    device: torch.device,
    *,
    diagnostic: str,
    overlap_work: Callable[[], object],
) -> None:
    """Launch CUDA P2P, execute independent local work, then join explicitly."""
    if device.type != "cuda":
        _wait_batched_p2p(operations, diagnostic=diagnostic)
        started = time.perf_counter()
        overlap_work()
        _MPS_P2P_STATS["overlap_work_seconds"] += time.perf_counter() - started
        return
    timeout = _p2p_timeout()
    key = (device.type, device.index)
    stream = _MPS_P2P_STREAMS.get(key)
    if stream is None:
        stream = get_platform_runtime(device.type).stream(device)
        _MPS_P2P_STREAMS[key] = stream
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        _MPS_P2P_STATS["physical_message_count"] += len(operations)
        requests = dist.batch_isend_irecv(operations)
    work_started = time.perf_counter()
    try:
        overlap_work()
    finally:
        _MPS_P2P_STATS["overlap_work_seconds"] += time.perf_counter() - work_started
        wait_started = time.perf_counter()
        try:
            for request in requests:
                try:
                    completed = request.wait(timeout=timeout)
                except Exception as exc:
                    raise RuntimeError(
                        f"MPS P2P failed: {diagnostic}; {_p2p_state()}; cause={exc}"
                    ) from exc
                if completed is False:
                    raise RuntimeError(f"MPS P2P timeout: {diagnostic}; {_p2p_state()}")
        finally:
            current.wait_stream(stream)
            _MPS_P2P_STATS["host_wait_seconds"] += time.perf_counter() - wait_started


def warmup_mps_neighbor_communicators(device: torch.device) -> float:
    """Collectively initialize every adjacent-rank P2P route before timing."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return 0.0
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.distributed_c10d._get_default_group()
    key = (id(group), world, rank, str(device))
    if key in _WARMED_MPS_NEIGHBORS:
        return 0.0
    started = time.perf_counter()
    operations: list[dist.P2POp] = []
    buffers: list[torch.Tensor] = []
    for peer in (rank - 1, rank + 1):
        if not 0 <= peer < world:
            continue
        send = torch.tensor([rank], dtype=torch.int64, device=device)
        recv = torch.empty_like(send)
        tag = 9_400_000 + min(rank, peer)
        buffers.extend((send, recv))
        operations.extend(
            (
                dist.P2POp(dist.isend, send, peer, group, tag),
                dist.P2POp(dist.irecv, recv, peer, group, tag),
            )
        )
    _run_batched_p2p(
        operations,
        device,
        diagnostic=f"phase=communicator_warmup,peers={tuple((rank - 1, rank + 1))}",
    )
    if device.type == "cuda":
        get_platform_runtime(device.type).synchronize(device)
    _WARMED_MPS_NEIGHBORS.add(key)
    return time.perf_counter() - started


def _send_tensor_p2p(
    tensor: torch.Tensor, *, dst: int, sequence: int | None = None
) -> None:
    tensor = tensor.contiguous()
    _MPS_P2P_STATS["logical_tensor_count"] += 1
    _MPS_P2P_STATS["logical_payload_bytes"] += tensor.numel() * tensor.element_size()
    descriptor = torch.tensor(
        ((-1 if sequence is None else int(sequence)), *tuple(tensor.shape)),
        dtype=torch.int64,
        device=tensor.device,
    )
    _MPS_P2P_STATS["descriptor_message_count"] += 1
    _run_batched_p2p(
        [
            dist.P2POp(dist.isend, descriptor, dst),
            dist.P2POp(dist.isend, tensor.reshape(-1), dst),
        ],
        tensor.device,
        diagnostic=(
            f"direction=send,peer={dst},sequence={sequence},"
            f"tensor_shape={tuple(tensor.shape)}"
        ),
    )


def _recv_tensor_p2p(
    *, src: int, reference: torch.Tensor, sequence: int | None = None
) -> torch.Tensor:
    descriptor = torch.empty(
        (reference.ndim + 1,), dtype=torch.int64, device=reference.device
    )
    _run_batched_p2p(
        [dist.P2POp(dist.irecv, descriptor, src)],
        reference.device,
        diagnostic=(
            f"direction=receive_descriptor,peer={src},sequence={sequence},"
            f"tensor_shape={tuple(reference.shape)}"
        ),
    )
    values = descriptor.detach().cpu().tolist()
    actual_sequence = int(values[0])
    sequence_mismatch = sequence is not None and actual_sequence != int(sequence)
    shape_tuple = tuple(int(value) for value in values[1:])
    flat = torch.empty(
        int(__import__("math").prod(shape_tuple)),
        dtype=reference.dtype,
        device=reference.device,
    )
    _run_batched_p2p(
        [dist.P2POp(dist.irecv, flat, src)],
        reference.device,
        diagnostic=(
            f"direction=receive_payload,peer={src},sequence={sequence},"
            f"actual_sequence={actual_sequence},tensor_shape={shape_tuple}"
        ),
    )
    if sequence_mismatch:
        raise RuntimeError(
            f"MPS P2P sequence mismatch from peer {src}: expected {sequence}, "
            f"received {actual_sequence}; payload drained before failure"
        )
    return flat.reshape(shape_tuple)


def _static_descriptor_key(
    peer: int, sequence: int, direction: int
) -> tuple[int, int, int]:
    return (int(peer), int(sequence), int(direction))


def _send_tensor_static_p2p(
    tensor: torch.Tensor,
    *,
    dst: int,
    sequence: int,
    expected_shape: Sequence[int],
    shape_generation: int,
) -> None:
    """Send payload-only after one descriptor validation for a shape generation."""
    shape = tuple(int(value) for value in expected_shape)
    if tuple(tensor.shape) != shape:
        raise RuntimeError(
            f"MPS static P2P shape mismatch before send: expected {shape}, received {tuple(tensor.shape)}"
        )
    key = _static_descriptor_key(dst, sequence, 0)
    signature = (int(shape_generation), shape, tensor.dtype)
    cached = _MPS_STATIC_DESCRIPTOR_CACHE.get(key)
    if cached != signature:
        if cached is not None:
            _MPS_P2P_STATS["descriptor_cache_invalidations"] += 1
        _MPS_P2P_STATS["descriptor_cache_misses"] += 1
        _send_tensor_p2p(tensor, dst=dst, sequence=sequence)
        _MPS_STATIC_DESCRIPTOR_CACHE[key] = signature
        return
    _MPS_P2P_STATS["descriptor_cache_hits"] += 1
    _MPS_P2P_STATS["logical_tensor_count"] += 1
    _MPS_P2P_STATS["logical_payload_bytes"] += tensor.numel() * tensor.element_size()
    _MPS_P2P_STATS["static_payload_message_count"] += 1
    _run_batched_p2p(
        [dist.P2POp(dist.isend, tensor.contiguous().reshape(-1), dst)],
        tensor.device,
        diagnostic=f"direction=static_send,peer={dst},sequence={sequence},generation={shape_generation}",
    )


def _recv_tensor_static_p2p(
    *,
    src: int,
    reference: torch.Tensor,
    sequence: int,
    expected_shape: Sequence[int],
    shape_generation: int,
) -> torch.Tensor:
    shape = tuple(int(value) for value in expected_shape)
    key = _static_descriptor_key(src, sequence, 1)
    signature = (int(shape_generation), shape, reference.dtype)
    cached = _MPS_STATIC_DESCRIPTOR_CACHE.get(key)
    if cached != signature:
        if cached is not None:
            _MPS_P2P_STATS["descriptor_cache_invalidations"] += 1
        _MPS_P2P_STATS["descriptor_cache_misses"] += 1
        descriptor = torch.empty(
            (len(shape) + 1,), dtype=torch.int64, device=reference.device
        )
        _run_batched_p2p(
            [dist.P2POp(dist.irecv, descriptor, src)],
            reference.device,
            diagnostic=(
                f"direction=static_receive_cold_descriptor,peer={src},"
                f"sequence={sequence},"
                f"generation={shape_generation},expected_shape={shape}"
            ),
        )
        values = tuple(int(value) for value in descriptor.detach().cpu().tolist())
        actual_sequence = values[0]
        actual_shape = tuple(values[1:])
        if any(dimension < 0 for dimension in actual_shape):
            raise RuntimeError(
                f"MPS static P2P descriptor from peer {src} contains a negative "
                f"dimension: {actual_shape}"
            )
        flat = torch.empty(
            int(__import__("math").prod(actual_shape)),
            dtype=reference.dtype,
            device=reference.device,
        )
        _run_batched_p2p(
            [dist.P2POp(dist.irecv, flat, src)],
            reference.device,
            diagnostic=(
                f"direction=static_receive_cold_payload,peer={src},"
                f"sequence={sequence},actual_sequence={actual_sequence},"
                f"actual_shape={actual_shape}"
            ),
        )
        if actual_sequence != int(sequence) or actual_shape != shape:
            raise RuntimeError(
                f"MPS static P2P descriptor mismatch from peer {src}: expected "
                f"sequence {sequence} and shape {shape}, received sequence "
                f"{actual_sequence} and shape {actual_shape}; payload drained before failure"
            )
        _MPS_STATIC_DESCRIPTOR_CACHE[key] = signature
        return flat.reshape(shape)
    _MPS_P2P_STATS["descriptor_cache_hits"] += 1
    flat = torch.empty(
        __import__("math").prod(shape), dtype=reference.dtype, device=reference.device
    )
    _MPS_P2P_STATS["static_payload_message_count"] += 1
    _run_batched_p2p(
        [dist.P2POp(dist.irecv, flat, src)],
        reference.device,
        diagnostic=f"direction=static_receive,peer={src},sequence={sequence},generation={shape_generation}",
    )
    return flat.reshape(shape)


def _send_tensor_batch_p2p(
    tensors: Sequence[torch.Tensor], *, dst: int, sequences: Sequence[int]
) -> None:
    """Send compatible tensors as one descriptor and one packed payload."""
    values = tuple(tensor.contiguous() for tensor in tensors)
    if not values or len(values) != len(sequences):
        raise ValueError("MPS tensor batch requires matching non-empty sequences")
    device, dtype = values[0].device, values[0].dtype
    if any(value.device != device or value.dtype != dtype for value in values):
        raise ValueError("MPS tensor batch requires one device and dtype")
    max_ndim = max(value.ndim for value in values)
    descriptor_values = [len(values), max_ndim]
    for sequence, value in zip(sequences, values):
        descriptor_values.extend((int(sequence), value.ndim, *value.shape))
        descriptor_values.extend((-1,) * (max_ndim - value.ndim))
    descriptor = torch.tensor(descriptor_values, dtype=torch.int64, device=device)
    _MPS_P2P_STATS["descriptor_message_count"] += 1
    elements = sum(value.numel() for value in values)
    payload = _pooled_p2p_buffer(device=device, dtype=dtype, elements=elements)
    offset = 0
    for value in values:
        count = value.numel()
        payload[offset : offset + count].copy_(value.reshape(-1))
        offset += count
    _MPS_P2P_STATS["logical_tensor_count"] += len(values)
    _MPS_P2P_STATS["logical_payload_bytes"] += payload.numel() * payload.element_size()
    _run_batched_p2p(
        [
            dist.P2POp(dist.isend, descriptor, dst),
            dist.P2POp(dist.isend, payload, dst),
        ],
        device,
        diagnostic=f"direction=batch_send,peer={dst},sequences={tuple(sequences)}",
    )


def _recv_tensor_batch_p2p(
    *,
    src: int,
    references: Sequence[torch.Tensor],
    sequences: Sequence[int],
    validate_shapes: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Receive a packed tensor batch and fail after draining on mismatch."""
    refs = tuple(references)
    if not refs or len(refs) != len(sequences):
        raise ValueError("MPS tensor batch requires matching non-empty sequences")
    device, dtype = refs[0].device, refs[0].dtype
    if any(value.device != device or value.dtype != dtype for value in refs):
        raise ValueError("MPS tensor batch requires one device and dtype")
    max_ndim = max(value.ndim for value in refs)
    descriptor = torch.empty(
        2 + len(refs) * (2 + max_ndim), dtype=torch.int64, device=device
    )
    _run_batched_p2p(
        [dist.P2POp(dist.irecv, descriptor, src)],
        device,
        diagnostic=f"direction=batch_receive_descriptor,peer={src}",
    )
    data = tuple(int(value) for value in descriptor.detach().cpu().tolist())
    count, actual_max_ndim = data[:2]
    mismatch = count != len(refs) or actual_max_ndim != max_ndim
    shapes = []
    actual_sequences = []
    cursor = 2
    for _ in refs:
        actual_sequence, ndim = data[cursor : cursor + 2]
        cursor += 2
        shape = tuple(data[cursor : cursor + max_ndim][:ndim])
        cursor += max_ndim
        actual_sequences.append(actual_sequence)
        shapes.append(shape)
    mismatch = mismatch or tuple(actual_sequences) != tuple(int(v) for v in sequences)
    if validate_shapes:
        mismatch = mismatch or tuple(shapes) != tuple(
            tuple(value.shape) for value in refs
        )
    elements = sum(int(__import__("math").prod(shape)) for shape in shapes)
    pooled = _pooled_p2p_buffer(device=device, dtype=dtype, elements=elements)
    _run_batched_p2p(
        [dist.P2POp(dist.irecv, pooled, src)],
        device,
        diagnostic=f"direction=batch_receive_payload,peer={src},sequences={tuple(sequences)}",
    )
    outputs = []
    offset = 0
    for shape in shapes:
        size = int(__import__("math").prod(shape))
        outputs.append(pooled[offset : offset + size].reshape(shape).clone())
        offset += size
    if mismatch:
        raise RuntimeError(
            f"MPS P2P batch mismatch from peer {src}: expected sequences "
            f"{tuple(sequences)}, received {tuple(actual_sequences)}; payload drained before failure"
        )
    return tuple(outputs)


def _send_tensor_async_p2p(tensor: torch.Tensor, *, dst: int) -> None:
    """Send a tensor shape and payload with asynchronous P2P calls."""

    tensor = tensor.contiguous()
    shape = torch.tensor(tuple(tensor.shape), dtype=torch.int64, device=tensor.device)
    requests = [
        dist.isend(shape, dst=dst),
        dist.isend(tensor.reshape(-1), dst=dst),
    ]
    for request in requests:
        request.wait()


def _recv_tensor_async_p2p(*, src: int, reference: torch.Tensor) -> torch.Tensor:
    """Receive an asynchronous tensor using a reference dtype and device."""

    shape = torch.empty((reference.ndim,), dtype=torch.int64, device=reference.device)
    dist.irecv(shape, src=src).wait()
    shape_tuple = tuple(int(value) for value in shape.detach().cpu().tolist())
    flat = torch.empty(
        int(torch.prod(shape).detach().cpu().item()),
        dtype=reference.dtype,
        device=reference.device,
    )
    dist.irecv(flat, src=src).wait()
    return flat.reshape(shape_tuple)


__all__ = (
    "clear_mps_static_descriptor_cache",
    "mps_p2p_stats",
    "mps_static_descriptor_cache_entries",
    "reset_mps_p2p_stats",
    "warmup_mps_neighbor_communicators",
)
