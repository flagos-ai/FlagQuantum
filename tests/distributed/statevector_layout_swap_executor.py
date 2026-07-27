"""Two-rank NCCL correctness check for persistent statevector bit swaps."""

import os

import torch
import torch.distributed as dist

from flagquantum.runtime.backends.statevector.layout import (
    distributed_swap_rank_local_bits,
)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    try:
        initial = torch.arange(rank, 8, 2, dtype=torch.float32, device=device).to(
            torch.complex64
        )[None, :]
        swapped, count, byte_count = distributed_swap_rank_local_bits(
            initial,
            rank=rank,
            n_wires=3,
            rank_bits=1,
            local_physical_wire=0,
            sharded_physical_wire=2,
        )
        restored, inverse_count, inverse_bytes = distributed_swap_rank_local_bits(
            swapped,
            rank=rank,
            n_wires=3,
            rank_bits=1,
            local_physical_wire=0,
            sharded_physical_wire=2,
            tag=1,
        )
        expected = (
            torch.tensor([0, 2, 1, 3], dtype=torch.float32, device=device)
            if rank == 0
            else torch.tensor([4, 6, 5, 7], dtype=torch.float32, device=device)
        ).to(torch.complex64)[None, :]
        assert torch.equal(swapped, expected)
        assert torch.equal(restored, initial)
        assert (count, inverse_count) == (1, 1)
        assert (byte_count, inverse_bytes) == (16, 16)
        dist.barrier()
        if rank == 0:
            print("persistent-layout NCCL bit swap: PASS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
