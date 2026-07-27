"""torchrun smoke test for distributed MPS and tensor-network modes.

Run manually, for example:

    torchrun --standalone --nproc_per_node=2 tests/distributed/runtime_modes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _circuit() -> fq.Circuit:
    return fq.Circuit(4).h(0).cx(1, 2).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)


def main() -> None:
    world_size = _world_size()
    rank = _rank()

    dtn = fq.run_native(
        _circuit(),
        mode="distributed_tensor_network",
        world_size=world_size,
        distributed_executor="torch",
        backend="gloo",
        device="cpu",
        max_intermediate_size=8,
    )
    local_tn = fq.run_native(_circuit(), mode="tensor_network")
    if not torch.allclose(dtn.to_statevector(), local_tn.to_statevector(), atol=1e-6):
        raise AssertionError(
            "distributed tensor-network result does not match local tensor-network result"
        )
    torch.distributed.barrier()

    original_broadcast_object_list = torch.distributed.broadcast_object_list
    original_all_gather_object = torch.distributed.all_gather_object

    def _forbid_object_collective(*args: object, **kwargs: object) -> None:
        raise AssertionError("distributed MPS runtime used an object collective")

    torch.distributed.broadcast_object_list = _forbid_object_collective
    torch.distributed.all_gather_object = _forbid_object_collective
    try:
        dmps = fq.run_native(
            _circuit(),
            mode="distributed_mps",
            world_size=world_size,
            distributed_executor="torch",
            backend="gloo",
            device="cpu",
            max_bond=8,
            boundary_transport="auto",
        )
    finally:
        torch.distributed.broadcast_object_list = original_broadcast_object_list
        torch.distributed.all_gather_object = original_all_gather_object
    local_mps = fq.run_native(_circuit(), mode="mps", max_bond=8)
    distributed_mps_state = dmps.sharded_state.to_statevector()
    local_mps_state = local_mps.to_statevector()
    if not torch.allclose(distributed_mps_state, local_mps_state, atol=1e-6):
        maximum_error = float(
            torch.max(torch.abs(distributed_mps_state - local_mps_state))
        )
        dense_state = _circuit().state()
        overlap = torch.sum(distributed_mps_state.conj() * dense_state)
        raise AssertionError(
            "distributed MPS result does not match local MPS result; "
            f"rank={rank}, maximum_error={maximum_error}, "
            f"distributed_norm={float(torch.linalg.vector_norm(distributed_mps_state))}, "
            f"local_norm={float(torch.linalg.vector_norm(local_mps_state))}, "
            f"fidelity={float(torch.abs(overlap).square())}, "
            f"distributed_dense_error={float(torch.max(torch.abs(distributed_mps_state - dense_state)))}, "
            f"local_dense_error={float(torch.max(torch.abs(local_mps_state - dense_state)))}"
        )
    if not dmps.local_shard_tensors:
        raise AssertionError("distributed MPS rank has no assigned local shard tensors")

    print(
        f"rank={rank} world_size={world_size} "
        f"tn_tasks={dtn.summary()['tasks_by_rank']} "
        f"mps_wires={sorted(dmps.local_shard_tensors)} "
        f"mps_storage={dmps.summary()['storage']} "
        f"sharded_wires={dmps.sharded_state.summary()['local_tensor_wires'] if dmps.sharded_state else ()} "
        f"mps_tensor_sync={dmps.summary()['tensor_sync_count']} "
        f"mps_boundary_sync={dmps.summary()['boundary_sync_count']} "
        f"mps_full_sync={dmps.summary()['full_sync_count']} "
        f"mps_sync_bytes={dmps.summary()['sync_bytes']} "
        f"mps_boundary_transfer_bytes={dmps.summary()['boundary_transfer_bytes']} "
        f"mps_boundary_protocols={dmps.summary()['boundary_protocols']}"
    )
    fq.destroy_torch_distributed()


if __name__ == "__main__":
    main()
