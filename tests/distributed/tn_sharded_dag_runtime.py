"""End-to-end retained-shard-chain check for the distributed TN DAG executor."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch
import torch.distributed as dist

from flagquantum.runtime.executors.tensor_network.distributed_dag import (
    TN_DAG_VERSION,
    DistributedTNContractionDAG,
    DistributedTNContractionRecord,
    DistributedTNValueLayout,
    _dag_identity,
    shard_distributed_tn_value_layout,
)
from flagquantum.runtime.executors.tensor_network.distributed_execution import (
    execute_sharded_tn_contraction_dag,
)


def _layout(
    value_id: str,
    labels: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    world_size: int,
    producer_id: str | None = None,
) -> DistributedTNValueLayout:
    elements = 1
    for extent in shape:
        elements *= extent
    return DistributedTNValueLayout(
        value_id=value_id,
        producer_id=producer_id,
        labels=labels,
        shape=shape,
        dtype="torch.complex64",
        nbytes=elements * 8,
        semantics="replicated_small" if producer_id is None else "unique_owner",
        owner_ranks=tuple(range(world_size)) if producer_id is None else (0,),
    )


def _dag(world_size: int, dimension: int) -> DistributedTNContractionDAG:
    inner = dimension // 2
    raw = (
        _layout("input:a", (0, 1), (dimension, inner), world_size=world_size),
        _layout("input:b", (1, 2), (inner, dimension), world_size=world_size),
        _layout("input:d", (0, 2), (dimension, dimension), world_size=world_size),
        _layout("input:e", (0, 2), (dimension, dimension), world_size=world_size),
    )
    x = shard_distributed_tn_value_layout(
        _layout(
            "intermediate:0",
            (0, 2),
            (dimension, dimension),
            world_size=world_size,
            producer_id="contract:0",
        ),
        world_size=world_size,
        shard_label=0,
    )
    y = shard_distributed_tn_value_layout(
        _layout(
            "intermediate:1",
            (0, 2),
            (dimension, dimension),
            world_size=world_size,
            producer_id="contract:1",
        ),
        world_size=world_size,
        shard_label=2,
    )
    output = _layout(
        "intermediate:2",
        (),
        (),
        world_size=world_size,
        producer_id="contract:2",
    )
    operations = (
        DistributedTNContractionRecord(
            "contract:0",
            0,
            ("input:a", "input:b"),
            x.value_id,
            x.owner_ranks,
            x.labels,
            x.shape,
            1,
            dimension * dimension,
        ),
        DistributedTNContractionRecord(
            "contract:1",
            1,
            (x.value_id, "input:d"),
            y.value_id,
            y.owner_ranks,
            y.labels,
            y.shape,
            1,
            dimension * dimension,
        ),
        DistributedTNContractionRecord(
            "contract:2",
            2,
            (y.value_id, "input:e"),
            output.value_id,
            output.owner_ranks,
            output.labels,
            output.shape,
            1,
            1,
        ),
    )
    values = raw + (x, y, output)
    payload = {
        "version": TN_DAG_VERSION,
        "world_size": world_size,
        "objective": "memory",
        "small_tensor_replication_bytes": 4096,
        "values": tuple(asdict(value) for value in values),
        "operations": tuple(asdict(operation) for operation in operations),
        "communication_edges": (),
        "output_value_id": output.value_id,
        "planning_only": True,
    }
    return DistributedTNContractionDAG(
        version=TN_DAG_VERSION,
        identity=_dag_identity(payload),
        world_size=world_size,
        objective="memory",
        small_tensor_replication_bytes=4096,
        values=values,
        operations=operations,
        communication_edges=(),
        output_value_id=output.value_id,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--dimension", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(
        f"cuda:{local_rank}" if arguments.backend == "nccl" else "cpu"
    )
    if arguments.backend == "nccl":
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    else:
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dimension = arguments.dimension
    if dimension <= 0 or dimension % world_size:
        raise ValueError("dimension must be positive and divisible by world size")
    inner = dimension // 2
    generator = torch.Generator(device=device).manual_seed(20260729)
    inputs = {
        "input:a": torch.randn(
            dimension, inner, dtype=torch.complex64, generator=generator, device=device
        ),
        "input:b": torch.randn(
            inner, dimension, dtype=torch.complex64, generator=generator, device=device
        ),
        "input:d": torch.randn(
            dimension,
            dimension,
            dtype=torch.complex64,
            generator=generator,
            device=device,
        ),
        "input:e": torch.randn(
            dimension,
            dimension,
            dtype=torch.complex64,
            generator=generator,
            device=device,
        ),
    }
    result = execute_sharded_tn_contraction_dag(_dag(world_size, dimension), inputs)
    reference = torch.einsum(
        "ab,bc,ac,ac->",
        inputs["input:a"],
        inputs["input:b"],
        inputs["input:d"],
        inputs["input:e"],
    )
    torch.testing.assert_close(result.value, reference, atol=1e-5, rtol=1e-5)
    assert result.executed_operation_ids == (
        "contract:0",
        "contract:1",
        "contract:2",
    )
    assert result.sharded_value_ids == ("intermediate:0", "intermediate:1")
    assert result.collective_count == 1
    assert result.redistribution_count == 1
    assert result.redistributed_value_ids == ("intermediate:0",)
    assert result.redistribution_sent_bytes > 0
    assert result.redistribution_received_bytes > 0
    print(json.dumps({"rank": rank, "sharded_dag_passed": True, **result.summary()}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
