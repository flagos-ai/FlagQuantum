"""Eight-rank dynamic Cartesian-mesh redistribution validation."""

import json
import os

import torch
import torch.distributed as dist

from flagquantum.runtime.backends.tensor_network.distributed_dag import (
    DistributedTNValueLayout,
)
from flagquantum.runtime.backends.tensor_network.multi_axis_sharding import (
    partition_tn_tensor_for_multi_axis_shard,
    plan_multi_axis_tn_layout,
)
from flagquantum.runtime.backends.tensor_network.redistribution import (
    execute_multi_axis_tn_redistribution,
    plan_multi_axis_tn_redistribution,
)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError("dynamic multi-axis redistribution requires eight ranks")

    base = DistributedTNValueLayout(
        value_id="adjoint:dynamic",
        producer_id="reverse:dynamic",
        labels=(10, 11, 12, 13, 14, 15),
        shape=(2, 2, 2, 2, 2, 2),
        dtype="torch.complex64",
        nbytes=64 * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )
    source = plan_multi_axis_tn_layout(
        base, shard_labels=(10, 11, 12), world_size=world_size
    )
    destination = plan_multi_axis_tn_layout(
        base, shard_labels=(13, 14, 15), world_size=world_size
    )
    logical = torch.arange(64, device=device).reshape(base.shape).to(torch.complex64)
    local_source = partition_tn_tensor_for_multi_axis_shard(logical, source, rank=rank)
    plan = plan_multi_axis_tn_redistribution(source, destination)
    result = execute_multi_axis_tn_redistribution(
        local_source, source, destination, plan
    )
    expected = partition_tn_tensor_for_multi_axis_shard(logical, destination, rank=rank)
    torch.testing.assert_close(result.local_tensor, expected)
    assert result.local_tensor.numel() == 8
    assert result.sent_bytes == 56
    assert result.received_bytes == 56
    assert result.self_transfer_bytes == 8
    assert result.physical_message_count == 14
    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "dynamic_multi_axis_redistribution_passed": True,
                "plan_identity": plan.identity,
                "source_layout_identity": source.identity,
                "destination_layout_identity": destination.identity,
                "source_shard_labels": source.shard_labels,
                "destination_shard_labels": destination.shard_labels,
                "local_shape": tuple(result.local_tensor.shape),
                "sent_bytes": result.sent_bytes,
                "received_bytes": result.received_bytes,
                "self_transfer_bytes": result.self_transfer_bytes,
                "physical_message_count": result.physical_message_count,
                "full_logical_tensor_materialized_by_executor": False,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
