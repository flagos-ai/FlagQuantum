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


def main() -> None:
    world_size = _world_size()
    rank = _rank()
    circuit = fq.Circuit(4)
    circuit.h(0).cx(1, 2).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)

    dtn = circuit.run(
        mode="distributed_tensor_network",
        world_size=world_size,
        distributed_executor="torch",
        backend="gloo",
        device="cpu",
        max_intermediate_size=8,
    )
    local_tn = circuit.run(mode="tensor_network")
    if not torch.allclose(dtn.state(), local_tn.state(), atol=1e-6):
        raise AssertionError(
            "distributed tensor-network result does not match local tensor-network result"
        )

    fq.destroy_torch_distributed()

    dmps = circuit.run(
        mode="distributed_mps",
        world_size=world_size,
        distributed_executor="torch",
        backend="gloo",
        device="cpu",
        max_bond=8,
        boundary_transport="auto",
    )
    local_mps = circuit.run(mode="mps", max_bond=8)
    if not torch.allclose(dmps.to_statevector(), local_mps.to_statevector(), atol=1e-6):
        raise AssertionError("distributed MPS result does not match local MPS result")
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
