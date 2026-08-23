"""Eight-rank reverse VJP with replicated mesh dimensions and subgroup sums."""

import json
import os

import torch
import torch.distributed as dist

from flagquantum.runtime.backends.tensor_network import (
    DistributedTNMeshGroupCache,
    DistributedTNValueLayout,
    execute_partial_mesh_reverse_pair,
    execute_partial_mesh_tn_redistribution,
    partition_tn_tensor_for_partial_mesh,
    plan_partial_mesh_tn_layout,
    plan_partial_mesh_tn_redistribution,
)


def _layout(value_id, labels):
    shape = (2,) * len(labels)
    return DistributedTNValueLayout(
        value_id=value_id,
        producer_id="reverse:test",
        labels=labels,
        shape=shape,
        dtype="torch.complex64",
        nbytes=(2 ** len(labels)) * 8,
        semantics="unique_owner",
        owner_ranks=(0,),
    )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise RuntimeError("partial mesh reverse runtime requires eight ranks")

    generator = torch.Generator().manual_seed(29)
    left = torch.complex(
        torch.randn((2, 2, 2, 2), generator=generator),
        torch.randn((2, 2, 2, 2), generator=generator),
    ).to(device)
    right = torch.complex(
        torch.randn((2, 2, 2, 2), generator=generator),
        torch.randn((2, 2, 2, 2), generator=generator),
    ).to(device)
    output_cotangent = torch.complex(
        torch.randn((2, 2, 2, 2), generator=generator),
        torch.randn((2, 2, 2, 2), generator=generator),
    ).to(device)
    mesh_labels = (0, 2, 4)
    mesh_shape = (2, 2, 2)
    left_layout = plan_partial_mesh_tn_layout(
        _layout("left", (0, 1, 2, 3)),
        mesh_labels=mesh_labels,
        mesh_shape=mesh_shape,
    )
    right_layout = plan_partial_mesh_tn_layout(
        _layout("right", (2, 3, 4, 5)),
        mesh_labels=mesh_labels,
        mesh_shape=mesh_shape,
    )
    output_layout = plan_partial_mesh_tn_layout(
        _layout("output", (0, 1, 4, 5)),
        mesh_labels=mesh_labels,
        mesh_shape=mesh_shape,
    )
    with DistributedTNMeshGroupCache(
        mesh_labels=mesh_labels,
        mesh_shape=mesh_shape,
    ) as group_cache:
        result = execute_partial_mesh_reverse_pair(
            partition_tn_tensor_for_partial_mesh(
                output_cotangent, output_layout, rank=rank
            ),
            output_layout,
            partition_tn_tensor_for_partial_mesh(left, left_layout, rank=rank),
            left_layout,
            partition_tn_tensor_for_partial_mesh(right, right_layout, rank=rank),
            right_layout,
            group_cache=group_cache,
        )
        group_count = group_cache.group_count
    reference_left = torch.einsum("abef,cdef->abcd", output_cotangent, right.conj())
    reference_right = torch.einsum("abcd,abef->cdef", left.conj(), output_cotangent)
    torch.testing.assert_close(
        result.left_cotangent,
        partition_tn_tensor_for_partial_mesh(reference_left, left_layout, rank=rank),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        result.right_cotangent,
        partition_tn_tensor_for_partial_mesh(reference_right, right_layout, rank=rank),
        atol=1e-5,
        rtol=1e-5,
    )
    remesh_layout = plan_partial_mesh_tn_layout(
        _layout("left", (0, 1, 2, 3)),
        mesh_labels=(1, 3, 6),
        mesh_shape=mesh_shape,
    )
    remesh_plan = plan_partial_mesh_tn_redistribution(left_layout, remesh_layout)
    remeshed = execute_partial_mesh_tn_redistribution(
        result.left_cotangent,
        left_layout,
        remesh_layout,
        remesh_plan,
    )
    torch.testing.assert_close(
        remeshed.local_tensor,
        partition_tn_tensor_for_partial_mesh(reference_left, remesh_layout, rank=rank),
        atol=1e-5,
        rtol=1e-5,
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "partial_mesh_reverse_passed": True,
                "mesh_labels": mesh_labels,
                "left_partitioned_labels": left_layout.partitioned_mesh_labels,
                "left_replicated_labels": left_layout.replicated_mesh_labels,
                "right_partitioned_labels": right_layout.partitioned_mesh_labels,
                "right_replicated_labels": right_layout.replicated_mesh_labels,
                "reduced_mesh_labels": result.reduced_mesh_labels,
                "subgroup_collective_count": result.subgroup_collective_count,
                "subgroup_collective_bytes": result.subgroup_collective_bytes,
                "cached_mesh_group_count": group_count,
                "partial_remesh_passed": True,
                "partial_remesh_plan_identity": remesh_plan.identity,
                "partial_remesh_destination_labels": (remesh_layout.mesh_labels),
                "partial_remesh_canonical_source_rank_count": (
                    remesh_plan.canonical_source_rank_count
                ),
                "partial_remesh_sent_bytes": remeshed.sent_bytes,
                "partial_remesh_received_bytes": remeshed.received_bytes,
                "full_logical_tensor_materialized_by_executor": False,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
