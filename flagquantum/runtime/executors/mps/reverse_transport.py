"""Sequenced tensor transport and metadata schemas for MPS reverse mode."""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.profiler import record_function

from ....compute import get_platform_runtime
from ....core.ir import Instruction
from .records import MPSReverseContractError
from .state import RankOwnedMPSState
from .transport import (
    _recv_tensor_batch_p2p,
    _recv_tensor_p2p,
    _recv_tensor_static_p2p,
    _send_tensor_batch_p2p,
    _send_tensor_p2p,
    _send_tensor_static_p2p,
)


@dataclass
class ReverseLayerHaloPrefetch:
    requests: tuple[Any, ...]
    buffers: tuple[torch.Tensor, ...]
    received: dict[int, torch.Tensor]
    stream: torch.cuda.Stream | None
    message_count: int
    payload_bytes: int
    intra_node_payload_bytes: int
    inter_node_payload_bytes: int


_LAYER_HALO_STREAMS: dict[int | None, torch.cuda.Stream] = {}


def begin_reverse_layer_halo_prefetch(
    layer: Sequence[tuple[int, Instruction, int]],
    state: RankOwnedMPSState,
    global_shapes: Mapping[int, Sequence[int]],
) -> tuple[ReverseLayerHaloPrefetch | None, frozenset[int]]:
    """Launch disjoint layer-boundary halo sends before rank-local compute."""
    boundary = tuple(
        (index, left_wire, state.owner(left_wire), state.owner(left_wire + 1))
        for index, _, left_wire in layer
        if state.owner(left_wire) != state.owner(left_wire + 1)
    )
    indices = frozenset(index for index, _, _, _ in boundary)
    operations = []
    buffers = []
    received = {}
    peers = []
    for index, left_wire, left_owner, right_owner in boundary:
        if state.rank == right_owner:
            value = state.local_tensors[left_wire + 1].contiguous().reshape(-1)
            buffers.append(value)
            peers.append(left_owner)
            operations.append(dist.P2POp(dist.isend, value, left_owner))
        elif state.rank == left_owner:
            shape = tuple(int(value) for value in global_shapes[left_wire + 1])
            reference = next(iter(state.local_tensors.values()))
            value = torch.empty(
                math.prod(shape),
                dtype=reference.dtype,
                device=reference.device,
            )
            buffers.append(value)
            peers.append(right_owner)
            received[index] = value.reshape(shape)
            operations.append(dist.P2POp(dist.irecv, value, right_owner))
    if not operations:
        return None, indices
    device = next(iter(state.local_tensors.values())).device
    stream = None
    if device.type == "cuda":
        stream = _LAYER_HALO_STREAMS.get(device.index)
        if stream is None:
            stream = get_platform_runtime(device.type).stream(device)
            _LAYER_HALO_STREAMS[device.index] = stream
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            requests = tuple(dist.batch_isend_irecv(operations))
    else:
        requests = tuple(dist.batch_isend_irecv(operations))
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", state.world_size)))
    payload_bytes = sum(value.numel() * value.element_size() for value in buffers)
    intra_node_payload_bytes = sum(
        value.numel() * value.element_size()
        for value, peer in zip(buffers, peers)
        if state.rank // local_world_size == peer // local_world_size
    )
    return (
        ReverseLayerHaloPrefetch(
            requests=requests,
            buffers=tuple(buffers),
            received=received,
            stream=stream,
            message_count=len(operations),
            payload_bytes=payload_bytes,
            intra_node_payload_bytes=intra_node_payload_bytes,
            inter_node_payload_bytes=payload_bytes - intra_node_payload_bytes,
        ),
        indices,
    )


def finish_reverse_layer_halo_prefetch(
    prefetch: ReverseLayerHaloPrefetch | None,
) -> float:
    """Wait for a prefetched boundary layer and join its CUDA stream."""
    if prefetch is None:
        return 0.0
    timeout = timedelta(
        seconds=float(os.environ.get("FLAGQUANTUM_MPS_P2P_TIMEOUT_SECONDS", "120"))
    )
    started = time.perf_counter()
    with record_function("flagquantum::mps::p2p_layer_prefetch_wait"):
        for request in prefetch.requests:
            completed = request.wait(timeout=timeout)
            if completed is False:
                raise MPSReverseContractError("MPS layer halo prefetch timed out")
    if prefetch.stream is not None:
        torch.cuda.current_stream().wait_stream(prefetch.stream)
    return time.perf_counter() - started


def send_reverse_tensor(
    tensor: torch.Tensor, *, destination: int, sequence: int
) -> None:
    with record_function("flagquantum::mps::p2p_send"):
        _send_tensor_p2p(tensor, dst=destination, sequence=sequence)


def receive_reverse_tensor(
    reference: torch.Tensor, *, source: int, sequence: int
) -> torch.Tensor:
    try:
        with record_function("flagquantum::mps::p2p_recv"):
            return _recv_tensor_p2p(src=source, reference=reference, sequence=sequence)
    except RuntimeError as exc:
        raise MPSReverseContractError(str(exc)) from exc


def send_reverse_tensor_batch(
    tensors: Sequence[torch.Tensor], *, destination: int, sequences: Sequence[int]
) -> None:
    with record_function("flagquantum::mps::p2p_batch_send"):
        _send_tensor_batch_p2p(tensors, dst=destination, sequences=sequences)


def receive_reverse_tensor_batch(
    references: Sequence[torch.Tensor],
    *,
    source: int,
    sequences: Sequence[int],
    validate_shapes: bool = True,
) -> tuple[torch.Tensor, ...]:
    try:
        with record_function("flagquantum::mps::p2p_batch_recv"):
            return _recv_tensor_batch_p2p(
                src=source,
                references=references,
                sequences=sequences,
                validate_shapes=validate_shapes,
            )
    except RuntimeError as exc:
        raise MPSReverseContractError(str(exc)) from exc


def encode_reverse_record(
    payload: dict[str, Any], reference: torch.Tensor
) -> torch.Tensor:
    inputs = tuple(tuple(int(v) for v in shape) for shape in payload["input_shapes"])
    outputs = tuple(tuple(int(v) for v in shape) for shape in payload["output_shapes"])
    if (
        len(inputs) != 2
        or len(outputs) != 2
        or any(len(shape) != 4 for shape in inputs + outputs)
    ):
        raise MPSReverseContractError(
            "MPS record tensor schema requires two rank-4 tensors"
        )
    info = dict(payload.get("split_info", {}))
    values = [
        1.0,
        *map(float, inputs[0]),
        *map(float, inputs[1]),
        *map(float, outputs[0]),
        *map(float, outputs[1]),
        float(info.get("rank", -1)),
        float(info.get("original_rank", -1)),
        float(info.get("discarded_weight", 0.0)),
        float(info.get("singular_value_gap", "nan")),
    ]
    return torch.tensor(values, dtype=torch.float64, device=reference.device)


def decode_reverse_record(schema: torch.Tensor) -> dict[str, Any]:
    values = schema.detach().cpu().tolist()
    if int(values[0]) != 1:
        raise MPSReverseContractError("unsupported MPS record tensor schema version")
    info = {"discarded_weight": float(values[19])}
    if int(values[17]) >= 0:
        info["rank"] = int(values[17])
    if int(values[18]) >= 0:
        info["original_rank"] = int(values[18])
    if not math.isnan(values[20]):
        info["singular_value_gap"] = float(values[20])
    return {
        "input_shapes": (tuple(map(int, values[1:5])), tuple(map(int, values[5:9]))),
        "output_shapes": (
            tuple(map(int, values[9:13])),
            tuple(map(int, values[13:17])),
        ),
        "split_info": info,
    }


def broadcast_reverse_record(
    payload: dict[str, Any] | None, owner: int, reference: torch.Tensor
) -> dict[str, Any]:
    """Broadcast dynamic split metadata with one fixed tensor schema."""
    schema = torch.empty(21, dtype=torch.float64, device=reference.device)
    if dist.get_rank() == owner:
        assert payload is not None
        schema.copy_(encode_reverse_record(payload, reference))
    dist.broadcast(schema, src=owner)
    return decode_reverse_record(schema)


def all_reduce_reverse_layer_records(
    payloads: Mapping[int, dict[str, Any]],
    entries: Sequence[tuple[int, int]],
    reference: torch.Tensor,
) -> dict[int, dict[str, Any]]:
    """Exchange one disjoint layer's dynamic metadata in one collective."""
    if not entries:
        return {}
    schemas = torch.zeros(
        (len(entries), 21), dtype=torch.float64, device=reference.device
    )
    rank = dist.get_rank()
    for position, (instruction_index, owner) in enumerate(entries):
        if rank == owner:
            schemas[position].copy_(
                encode_reverse_record(payloads[instruction_index], reference)
            )
    dist.all_reduce(schemas, op=dist.ReduceOp.SUM)
    return {
        instruction_index: decode_reverse_record(schemas[position])
        for position, (instruction_index, _) in enumerate(entries)
    }


def static_shape_generation(shape: Sequence[int]) -> int:
    content = ",".join(str(int(value)) for value in shape).encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:7], "little")


def send_static_reverse_tensor(
    tensor: torch.Tensor, *, destination: int, sequence: int, shape: Sequence[int]
) -> None:
    with record_function("flagquantum::mps::p2p_static_send"):
        _send_tensor_static_p2p(
            tensor,
            dst=destination,
            sequence=sequence,
            expected_shape=shape,
            shape_generation=static_shape_generation(shape),
        )


def receive_static_reverse_tensor(
    reference: torch.Tensor, *, source: int, sequence: int, shape: Sequence[int]
) -> torch.Tensor:
    try:
        with record_function("flagquantum::mps::p2p_static_recv"):
            return _recv_tensor_static_p2p(
                src=source,
                reference=reference,
                sequence=sequence,
                expected_shape=shape,
                shape_generation=static_shape_generation(shape),
            )
    except RuntimeError as exc:
        raise MPSReverseContractError(str(exc)) from exc


__all__ = (
    "ReverseLayerHaloPrefetch",
    "begin_reverse_layer_halo_prefetch",
    "broadcast_reverse_record",
    "all_reduce_reverse_layer_records",
    "decode_reverse_record",
    "encode_reverse_record",
    "finish_reverse_layer_halo_prefetch",
    "receive_reverse_tensor",
    "receive_reverse_tensor_batch",
    "receive_static_reverse_tensor",
    "send_reverse_tensor",
    "send_reverse_tensor_batch",
    "send_static_reverse_tensor",
    "static_shape_generation",
)
