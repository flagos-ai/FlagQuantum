"""Distributed correctness/measurement check for cross-axis TN redistribution."""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from flagquantum.runtime.backends.tensor_network.distributed_dag import (
    DistributedTNValueLayout,
    shard_distributed_tn_value_layout,
)
from flagquantum.runtime.backends.tensor_network.redistribution import (
    execute_distributed_tn_redistribution,
    plan_distributed_tn_redistribution,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(
        f"cuda:{local_rank}" if arguments.backend == "nccl" else "cpu"
    )
    if arguments.backend == "nccl":
        torch.cuda.set_device(device)
        dist.init_process_group(arguments.backend, device_id=device)
    else:
        dist.init_process_group(arguments.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("TN redistribution runtime requires at least two ranks")
    if arguments.dimension <= 0 or arguments.dimension % world_size:
        raise ValueError("dimension must be positive and divisible by world size")
    if arguments.iterations <= 0 or arguments.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    dimension = arguments.dimension
    base = DistributedTNValueLayout(
        value_id="intermediate:runtime",
        producer_id="contract:runtime",
        labels=(10, 11),
        shape=(dimension, dimension),
        dtype="torch.complex64",
        nbytes=dimension * dimension * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = shard_distributed_tn_value_layout(
        base, world_size=world_size, shard_label=10
    )
    destination = shard_distributed_tn_value_layout(
        base, world_size=world_size, shard_label=11
    )
    source_shard = source.shards[rank]
    rows = torch.arange(source_shard.start, source_shard.stop, device=device).reshape(
        -1, 1
    )
    columns = torch.arange(base.shape[1], device=device).reshape(1, -1)
    local_source = (rows * dimension + columns).to(torch.complex64)
    plan = plan_distributed_tn_redistribution(source, destination)

    result = None
    elapsed_seconds = 0.0
    for iteration in range(arguments.warmup + arguments.iterations):
        dist.barrier()
        started = time.perf_counter()
        candidate = execute_distributed_tn_redistribution(
            local_source,
            source,
            destination,
            plan,
        )
        if arguments.backend == "nccl":
            torch.cuda.synchronize(device)
        iteration_seconds = time.perf_counter() - started
        if iteration >= arguments.warmup:
            elapsed_seconds += iteration_seconds
        result = candidate
    assert result is not None

    destination_shard = destination.shards[rank]
    expected_rows = torch.arange(base.shape[0], device=device).reshape(-1, 1)
    expected_columns = torch.arange(
        destination_shard.start, destination_shard.stop, device=device
    ).reshape(1, -1)
    expected = (expected_rows * dimension + expected_columns).to(torch.complex64)
    torch.testing.assert_close(result.local_tensor, expected)
    assert result.local_tensor.numel() < base.shape[0] * base.shape[1]
    assert result.sent_bytes > 0
    assert result.received_bytes > 0
    assert result.self_transfer_bytes > 0
    assert result.physical_message_count == 2 * (world_size - 1)
    mean_seconds = elapsed_seconds / arguments.iterations
    transferred_bytes = result.sent_bytes + result.received_bytes
    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "backend": arguments.backend,
                "device": str(device),
                "dimension": dimension,
                "iterations": arguments.iterations,
                "redistribution_passed": True,
                "local_shape": tuple(result.local_tensor.shape),
                "sent_bytes": result.sent_bytes,
                "received_bytes": result.received_bytes,
                "self_transfer_bytes": result.self_transfer_bytes,
                "physical_message_count": result.physical_message_count,
                "full_logical_tensor_materialized": False,
                "mean_seconds": mean_seconds,
                "rank_bidirectional_gib_per_second": (
                    transferred_bytes / mean_seconds / (1024**3)
                ),
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
